"""Decode stage export for TVM."""

import logging
import warnings

import torch

warnings.filterwarnings("ignore", module="torch._dynamo")
warnings.filterwarnings("ignore", module="torch._export")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
logging.getLogger("torch._export.non_strict_utils").setLevel(logging.ERROR)
from transformers.cache_utils import DynamicCache

from stage_common import flatten_kv_cache, get_text_model_dims


def rebuild_kv_cache(num_layers: int, past_kv_flat: tuple[torch.Tensor, ...]) -> DynamicCache:
    assert len(past_kv_flat) == num_layers * 2
    cache_data = [
        (past_kv_flat[i * 2], past_kv_flat[i * 2 + 1])
        for i in range(num_layers)
    ]
    return DynamicCache(ddp_cache_data=cache_data)


def make_4d_decode_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a 4D attention mask for decode (single query token).
    4D mask bypasses transformers' masking_utils and avoids unsupported bool & ops.
    
    attention_mask: (batch, kv_seq_len) - 2D padding mask
    Returns: (batch, 1, 1, kv_seq_len) float mask where 0.0=attend, -inf=ignore
    """
    # (batch, 1, 1, kv_seq) - decode attends to all previous positions
    mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    # Convert to additive mask: 0 -> -inf, 1 -> 0
    return (1.0 - mask_4d) * torch.finfo(dtype).min


class DecodeStageWrapper(torch.nn.Module):
    """Decode: 1 token + KV -> logits + new flattened KV."""

    def __init__(self, model, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, cache_position, *past_kv_flat):
        cache = rebuild_kv_cache(self.num_layers, past_kv_flat)
        # Create 4D mask to bypass transformers' masking_utils (avoids bool & op)
        mask_4d = make_4d_decode_mask(attention_mask, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=mask_4d,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        new_kv_flat = flatten_kv_cache(outputs.past_key_values)
        return outputs.logits, *new_kv_flat


def export_decode_stage_tvm(
    model,
    output_path,
    *,
    max_batch_size: int,
    max_seq_len: int,
):
    """Export decode stage to TVM via torch.export."""
    from tvm import relax
    from tvm.relax.frontend.torch import from_exported_program  # type: ignore

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    wrapper = DecodeStageWrapper(model, num_layers)
    wrapper.eval()

    assert max_seq_len > 1
    batch_size = 1
    seq_len = min(100, max_seq_len - 1)
    input_ids = torch.randint(0, 1000, (batch_size, 1))
    attention_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[seq_len]], dtype=torch.long)
    cache_position = torch.tensor([seq_len], dtype=torch.long)

    past_kv_flat = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        past_kv_flat.extend([k, v])

    print(f"  input_ids: {input_ids.shape} (1 new token)")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(f"  cache_position: {cache_position.shape}")
    print(f"  KV cache: {num_layers} layers x 2 x [{batch_size}, {num_kv_heads}, {seq_len}, {head_dim}]")

    # Static export: TVM's relax.add can't handle unrelated symbolic dims
    # from Dim.AUTO, and named Dim with derived relation (seq = kv_seq + 1)
    # triggers constraint violation in torch.export. Use static shapes for now.
    dynamic_shapes = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Exporting with torch.export...")
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            args=(input_ids, attention_mask, position_ids, cache_position, *past_kv_flat),
            strict=False,
            dynamic_shapes=dynamic_shapes,
        )

    print(f"  Converting to TVM Relax...")
    mod = from_exported_program(exported, keep_params_as_input=True)

    # Save Relax IR
    ir_path = output_path.with_suffix(".txt")
    with open(ir_path, "w") as f:
        f.write(str(mod))
    print(f"  Saved IR: {ir_path}")
    print(f"  Static shapes: batch=1, kv_seq={seq_len}")

    return mod, {}
