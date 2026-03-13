"""Prefill stage export for Qwen3-VL."""

import operator
import os

import torch

from stage_common import flatten_kv_cache, get_text_model_dims
from vision_static import StaticQwen3VLVision


def adapt_visual_for_prefill_export(model, image_grid_thw: torch.Tensor) -> None:
    model.model.visual = StaticQwen3VLVision(model.model.visual, image_grid_thw)
    model.model.visual.deepstack_visual_indexes = []
    model.model.visual.deepstack_merger_list = torch.nn.ModuleList()


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


def _make_tvm_custom_convert_map():
    from torch import fx
    from tvm import relax

    def _get_arg(arg, importer):
        if isinstance(arg, fx.Node):
            return importer.env[arg]
        return arg

    def _to_relax_expr(value, *, importer, ref_expr):
        if isinstance(value, relax.Expr):
            return value
        return relax.const(value, dtype=ref_expr.struct_info.dtype)

    def _eq(node, importer):
        lhs = _get_arg(node.args[0], importer)
        rhs = _get_arg(node.args[1], importer)
        if isinstance(lhs, relax.Expr) and not isinstance(rhs, relax.Expr):
            rhs = _to_relax_expr(rhs, importer=importer, ref_expr=lhs)
        if isinstance(rhs, relax.Expr) and not isinstance(lhs, relax.Expr):
            lhs = _to_relax_expr(lhs, importer=importer, ref_expr=rhs)
        if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
            return importer.block_builder.emit(relax.op.equal(lhs, rhs))
        return operator.eq(lhs, rhs)

    def _prod_dim_int(node, importer):
        args = importer.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return importer.block_builder.emit(relax.op.prod(x, dim, keepdims=keepdim))

    def _masked_scatter(node, importer):
        x = importer.env[node.args[0]]
        mask = importer.env[node.args[1]]
        source = importer.env[node.args[2]]
        ndim = len(mask.struct_info.shape)
        if ndim == 1:
            index = importer.block_builder.emit(relax.op.cumsum(mask, 0, dtype="int32"))
            index = importer.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            gathered_source = importer.block_builder.emit(relax.op.take(source, index, axis=0))
        else:
            f_mask = importer.block_builder.emit(relax.op.reshape(mask, [-1]))
            index = importer.block_builder.emit(relax.op.cumsum(f_mask, 0, dtype="int32"))
            index = importer.block_builder.emit(relax.op.subtract(index, relax.const(1, "int32")))
            source_shape = [-1] + [s for idx, s in enumerate(source.struct_info.shape) if idx >= ndim]
            f_source = importer.block_builder.emit(relax.op.reshape(source, source_shape))
            gathered_source = importer.block_builder.emit(relax.op.take(f_source, index, axis=0))
            gathered_source = importer.block_builder.emit(
                relax.op.reshape(gathered_source, x.struct_info.shape)
            )
        if ndim != len(x.struct_info.shape):
            mask = importer.block_builder.emit(relax.op.broadcast_to(mask, x.struct_info.shape))
        return importer.block_builder.emit(relax.op.where(mask, gathered_source, x))

    return {
        "eq": _eq,
        "eq.default": _eq,
        "masked_scatter.default": _masked_scatter,
        "prod.dim_int": _prod_dim_int,
    }


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

    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        custom_convert_map=_make_tvm_custom_convert_map(),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(str(mod))
    print(f"  Saved IR: {output_path.with_suffix('.txt')}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return mod, {}
