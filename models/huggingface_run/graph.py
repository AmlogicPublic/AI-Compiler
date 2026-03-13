"""生成模型的4个图: 模块树TXT, 模块图SVG, Forward/Backward graph SVG"""
import os
import gc
import torch

from shared import MODELS, load_model

BATCH, SEQ = 1, 16
QWEN_IMG, QWEN_PATCH, QWEN_TEMPORAL, QWEN_MERGE = 64, 16, 2, 2


def make_inputs(model, config):
    """构建模型输入"""
    cls_name = model.__class__.__name__
    if "Qwen3VL" in cls_name:
        return _make_qwen_vl_inputs(config)
    if "UNet2DCondition" in cls_name:
        return _make_unet_inputs(config)
    return {"input_ids": torch.randint(0, 1000, (BATCH, SEQ))}


def _make_unet_inputs(config):
    """UNet 输入 (latent + timestep + encoder_hidden_states)"""
    in_channels = getattr(config, 'in_channels', 4)
    sample_size = getattr(config, 'sample_size', 64)
    cross_attn_dim = getattr(config, 'cross_attention_dim', 1024)
    
    latent = torch.randn(BATCH, in_channels, sample_size, sample_size)
    timestep = torch.tensor([999])
    encoder_hidden = torch.randn(BATCH, 77, cross_attn_dim)
    return [latent, timestep, encoder_hidden]


def _make_qwen_vl_inputs(config):
    """Qwen3-VL 输入 (文本 + 视觉)"""
    img_tok = getattr(config, 'image_token_id', 151655)
    start_tok = getattr(config, 'vision_start_token_id', 151652)
    end_tok = getattr(config, 'vision_end_token_id', 151653)
    
    h_m = w_m = QWEN_IMG // QWEN_PATCH // QWEN_MERGE
    t_m = QWEN_TEMPORAL // QWEN_TEMPORAL
    n_patches = t_m * h_m * w_m
    channels = 3 * QWEN_TEMPORAL * QWEN_PATCH * QWEN_PATCH
    
    pixel_values = torch.randn(n_patches, channels)
    image_grid_thw = torch.tensor([[t_m, h_m, w_m]])
    
    n_text = SEQ - 3
    assert n_text > 0
    n_before, n_after = n_text // 2, n_text - n_text // 2
    
    input_ids = torch.cat([
        torch.randint(0, 1000, (BATCH, n_before)),
        torch.full((BATCH, 1), start_tok, dtype=torch.long),
        torch.full((BATCH, 1), img_tok, dtype=torch.long),
        torch.full((BATCH, 1), end_tok, dtype=torch.long),
        torch.randint(0, 1000, (BATCH, n_after)),
    ], dim=1)
    
    return {"input_ids": input_ids, "pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def _get_output(model, inputs):
    """获取模型输出张量"""
    if isinstance(inputs, list):
        out = model(*inputs)
    else:
        out = model(**inputs)
    if hasattr(out, 'sample'):
        return out.sample
    if hasattr(out, 'logits'):
        return out.logits
    return out.last_hidden_state


def gen_arch_txt(model, config, path, depth):
    """模块树 → TXT"""
    if os.path.exists(path):
        print(f"  [1/4] 模块树: {path} (已存在)")
        return
    from torchinfo import summary
    inputs = make_inputs(model, config)
    cols = ["input_size", "output_size", "num_params", "params_percent"]
    if "Bert" not in model.__class__.__name__:
        cols.append("mult_adds")
    stats = summary(model, input_data=inputs, depth=depth, col_names=cols,
                    col_width=16, row_settings=["var_names"], verbose=0)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(stats))
    print(f"  [1/4] 模块树: {path}")


def gen_module_svg(model, config, path, depth):
    """模块图 → SVG"""
    svg_path = path + ".svg"
    if os.path.exists(svg_path):
        print(f"  [2/4] 模块图: {svg_path} (已存在)")
        return
    from torchview import draw_graph
    inputs = make_inputs(model, config)
    
    try:
        graph = draw_graph(model, input_data=inputs, depth=depth,
                           hide_module_functions=True, hide_inner_tensors=True,
                           roll=True, show_shapes=True, strict=False)
        for attr in ['graph_attr', 'node_attr', 'edge_attr']:
            getattr(graph.visual_graph, attr)['fontname'] = 'Arial'
        graph.visual_graph.render(path, format='svg', cleanup=True)
        print(f"  [2/4] 模块图: {svg_path}")
    except KeyError as e:
        print(f"  [2/4] 模块图: 跳过 (torchview bug: {e})")


def gen_autograd_svg(model, config, path, backward=False):
    """Forward/Backward graph → SVG"""
    svg_path = path + ".svg"
    tag = "Backward" if backward else "Forward"
    idx = 4 if backward else 3
    if os.path.exists(svg_path):
        print(f"  [{idx}/4] {tag} graph: {svg_path} (已存在)")
        return
    from torchviz import make_dot
    inputs = make_inputs(model, config)
    target = _get_output(model, inputs)
    if backward:
        target = target.sum()
    dot = make_dot(target, params=dict(model.named_parameters()),
                   show_attrs=False, show_saved=backward)
    dot.attr(rankdir='TB')
    dot.render(path, format="svg", cleanup=True)
    print(f"  [{idx}/4] {tag} graph: {svg_path}")


def get_depth(model):
    """根据参数量设置 depth"""
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 200_000_000:      # < 200M
        return float('inf')
    if n_params < 1_000_000_000:    # < 1B
        return 5
    if n_params < 3_000_000_000:    # < 3B
        return 4
    return 3


def visualize(name, out_dir="output"):
    """生成单个模型的所有可视化"""
    path = os.path.join(out_dir, name)
    os.makedirs(path, exist_ok=True)
    
    print(f"\n[{name}] ({MODELS[name]})")
    
    model, config, _ = load_model(name, load_tokenizer=False)
    depth = get_depth(model)
    
    gen_arch_txt(model, config, os.path.join(path, f"architecture_{depth}.txt"), depth)
    gen_module_svg(model, config, os.path.join(path, f"module_graph_{depth}"), depth)
    gen_autograd_svg(model, config, os.path.join(path, "forward_graph"), backward=False)
    gen_autograd_svg(model, config, os.path.join(path, "backward_graph"), backward=True)
    
    del model
    gc.collect()


if __name__ == "__main__":
    print(f"目标模型: {list(MODELS.keys())}")
    
    for name in MODELS:
        visualize(name)
    
    print("\n完成!")
