"""Decode stage export."""

import torch
from transformers.cache_utils import DynamicCache

from stage_common import flatten_kv_cache, get_text_model_dims


def rebuild_kv_cache(num_layers: int, past_kv_flat: tuple[torch.Tensor, ...]) -> DynamicCache:
    assert len(past_kv_flat) == num_layers * 2
    cache_data = [
        (past_kv_flat[i * 2], past_kv_flat[i * 2 + 1])
        for i in range(num_layers)
    ]
    return DynamicCache(ddp_cache_data=cache_data)


class DecodeStageWrapper(torch.nn.Module):
    """Decode: 1 token + KV -> logits + new flattened KV."""

    def __init__(self, model, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, cache_position, *past_kv_flat):
        cache = rebuild_kv_cache(self.num_layers, past_kv_flat)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        new_kv_flat = flatten_kv_cache(outputs.past_key_values)
        return outputs.logits, *new_kv_flat


def export_decode_stage_mlir(
    model,
    decode_mlir_path,
    *,
    max_batch_size: int,
    max_seq_len: int,
):
    import iree.turbine.aot as aot
    from torch.export import Dim

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    wrapper = DecodeStageWrapper(model, num_layers)
    wrapper.eval()

    batch_size = 1
    seq_len = 100
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

    batch_dim = Dim.AUTO(min=1, max=max_batch_size)
    seq_dim = Dim.AUTO(min=1, max=max_seq_len)
    kv_seq_dim = Dim.AUTO(min=1, max=max_seq_len - 1)

    kv_dynamic_shapes = [{0: batch_dim, 2: kv_seq_dim} for _ in range(num_layers * 2)]
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim},
        {},
        tuple(kv_dynamic_shapes),
    )

    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask, position_ids, cache_position, *past_kv_flat),
        strict_export=True,
        dynamic_shapes=dynamic_shapes,
    )

    decode_mlir_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(decode_mlir_path))
    print(f"  Saved: {decode_mlir_path}")
    print(f"  Dynamic: batch=[1,{max_batch_size}], seq=[1,{max_seq_len}]")
    return True
