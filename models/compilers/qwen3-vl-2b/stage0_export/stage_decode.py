"""Decode stage export for Qwen3-VL."""

import operator

import torch
from transformers.cache_utils import DynamicCache

from stage_common import flatten_kv_cache, get_text_model_dims


def rebuild_kv_cache(num_layers: int, past_kv_flat: tuple[torch.Tensor, ...]) -> DynamicCache:
    assert len(past_kv_flat) == num_layers * 2
    cache_data = [(past_kv_flat[i * 2], past_kv_flat[i * 2 + 1]) for i in range(num_layers)]
    return DynamicCache(ddp_cache_data=cache_data)


class DecodeStageWrapper(torch.nn.Module):
    def __init__(self, model, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, *past_kv_flat):
        cache = rebuild_kv_cache(self.num_layers, past_kv_flat)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return outputs.logits, *flatten_kv_cache(outputs.past_key_values)


def _make_custom_convert_map():
    from torch import fx
    from tvm import relax

    def _get_arg(arg, importer):
        if isinstance(arg, fx.Node):
            return importer.env[arg]
        return arg

    def _is_tensor_var(x):
        if not isinstance(x, relax.Var):
            return False
        return isinstance(getattr(x, "struct_info", None), relax.TensorStructInfo)

    def _add(node, importer):
        lhs = _get_arg(node.args[0], importer)
        rhs = _get_arg(node.args[1], importer)
        if _is_tensor_var(lhs) or _is_tensor_var(rhs):
            if not isinstance(lhs, relax.Expr):
                lhs = relax.const(lhs, dtype=rhs.struct_info.dtype)
            if not isinstance(rhs, relax.Expr):
                rhs = relax.const(rhs, dtype=lhs.struct_info.dtype)
            return importer.block_builder.emit(relax.op.add(lhs, rhs))
        return operator.add(lhs, rhs)

    return {"add": _add}


def _build_decode_inputs(num_layers: int, num_kv_heads: int, head_dim: int):
    batch_size = 1
    seq_len = 100
    input_ids = torch.randint(0, 1000, (batch_size, 1))
    attention_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[seq_len]], dtype=torch.long)

    past_kv_flat = []
    for _ in range(num_layers):
        past_kv_flat.extend(
            [
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
            ]
        )
    return input_ids, attention_mask, position_ids, past_kv_flat


def export_decode_stage_mlir(model, decode_mlir_path, *, max_batch_size: int, max_seq_len: int):
    import iree.turbine.aot as aot
    from torch.export import Dim

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    wrapper = DecodeStageWrapper(model, num_layers)
    wrapper.eval()

    input_ids, attention_mask, position_ids, past_kv_flat = _build_decode_inputs(num_layers, num_kv_heads, head_dim)

    batch_dim = Dim.AUTO(min=1, max=max_batch_size)
    seq_dim = Dim.AUTO(min=1, max=max_seq_len)
    kv_seq_dim = Dim.AUTO(min=1, max=max_seq_len - 1)

    kv_dynamic_shapes = [{0: batch_dim, 2: kv_seq_dim} for _ in range(num_layers * 2)]
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim},
        tuple(kv_dynamic_shapes),
    )

    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask, position_ids, *past_kv_flat),
        strict_export=True,
        dynamic_shapes=dynamic_shapes,
    )

    decode_mlir_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(decode_mlir_path))
    print(f"  Saved: {decode_mlir_path}")
    print(f"  Dynamic: batch=[1,{max_batch_size}], seq=[1,{max_seq_len}]")
    return True


def export_decode_stage_tvm(model, output_path, *, max_batch_size: int, max_seq_len: int):
    from torch.export import Dim
    from tvm.relax.frontend.torch import from_exported_program  # type: ignore

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    wrapper = DecodeStageWrapper(model, num_layers)
    wrapper.eval()

    input_ids, attention_mask, position_ids, past_kv_flat = _build_decode_inputs(num_layers, num_kv_heads, head_dim)

    batch_dim = Dim.AUTO(min=1, max=max_batch_size)
    seq_dim = Dim.AUTO(min=1, max=max_seq_len)
    kv_seq_dim = Dim.AUTO(min=1, max=max_seq_len - 1)
    kv_dynamic_shapes = [{0: batch_dim, 2: kv_seq_dim} for _ in range(num_layers * 2)]
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim},
        tuple(kv_dynamic_shapes),
    )

    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            args=(input_ids, attention_mask, position_ids, *past_kv_flat),
            strict=False,
            dynamic_shapes=dynamic_shapes,
        )

    mod = from_exported_program(
        exported,
        keep_params_as_input=True,
        custom_convert_map=_make_custom_convert_map(),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(str(mod))
    print(f"  Saved IR: {output_path.with_suffix('.txt')}")
    print(f"  Dynamic: batch=[1,{max_batch_size}], seq=[1,{max_seq_len}]")
    return mod, {}
