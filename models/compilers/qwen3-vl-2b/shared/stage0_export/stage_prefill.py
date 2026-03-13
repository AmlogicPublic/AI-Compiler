"""Prefill stage export for Qwen3-VL."""

import os

import torch

from stage_common import flatten_kv_cache, get_text_model_dims
from vision_static import StaticQwen3VLVision


def adapt_visual_for_prefill_export(model, image_grid_thw: torch.Tensor) -> None:
    model.model.visual = StaticQwen3VLVision(model.model.visual, image_grid_thw)


class PrefillStageWrapper(torch.nn.Module):
    def __init__(self, model, image_grid_thw: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("image_grid_thw", image_grid_thw)
        adapt_visual_for_prefill_export(self.model, image_grid_thw)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=self.image_grid_thw,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, *flatten_kv_cache(outputs.past_key_values)


def export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path):
    import iree.turbine.aot as aot

    os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"
    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]
    pixel_values = prefill_example_inputs["pixel_values"]
    image_grid_thw = prefill_example_inputs["image_grid_thw"]

    wrapper = PrefillStageWrapper(model, image_grid_thw)
    wrapper.eval()
    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask, pixel_values),
        strict_export=True,
    )

    prefill_mlir_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(prefill_mlir_path))
    print(f"  Saved: {prefill_mlir_path}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return True


def export_prefill_stage_tvm(model, prefill_example_inputs, output_path):
    from tvm.relax.frontend.torch import from_exported_program  # type: ignore

    os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"
    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]
    pixel_values = prefill_example_inputs["pixel_values"]
    image_grid_thw = prefill_example_inputs["image_grid_thw"]

    wrapper = PrefillStageWrapper(model, image_grid_thw)
    wrapper.eval()

    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            args=(input_ids, attention_mask, pixel_values),
            strict=False,
        )

    # ==========================================================================
    # KNOWN ISSUE: TVM export fails on aten.item.default
    # ==========================================================================
    # Root cause:
    #   image_grid_thw.prod(dim=-1) // 4 -> unbind -> getitem -> .item()
    #   The .item() returns a Python scalar (196) used by split_with_sizes.
    #
    # TVM's _item implementation uses relax.op.take(x, 0, axis=0), but the
    # input is already a 0-dim tensor (scalar), so axis=0 is out of bounds.
    #
    # Attempted solutions (all failed):
    #   1. custom_convert_map for aten.item - TVM expects relax.Expr return,
    #      but split_with_sizes needs Python int
    #   2. torch.fx.experimental.const_fold.split_const_subgraphs - doesn't
    #      fold buffer placeholders, only module attributes
    #   3. Manual graph rewrite to inline buffer and fold - breaks
    #      ExportedProgram.graph_signature consistency
    #   4. Replace item node users with constant - split_with_sizes args is
    #      nested list [item], simple arg replacement doesn't work
    #
    # Potential solutions (not yet attempted):
    #   - Modify transformers model code to avoid .item() call
    #   - Patch TVM's _item to handle 0-dim tensors
    #   - Use torch.compile with different backend
    # ==========================================================================

    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(str(mod))
    print(f"  Saved IR: {output_path.with_suffix('.txt')}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return mod, {}
