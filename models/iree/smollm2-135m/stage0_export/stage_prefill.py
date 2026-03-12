"""Prefill stage export."""

import torch

from stage_common import flatten_kv_cache, get_text_model_dims


def make_4d_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a 4D causal attention mask from a 2D padding mask.
    4D mask bypasses transformers' masking_utils and avoids unsupported bool & ops.
    
    Returns: (batch, 1, seq_len, seq_len) float mask where 0.0=attend, -inf=ignore
    """
    batch_size, seq_len = attention_mask.shape
    # Causal mask: lower triangular
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=attention_mask.device))
    # Expand to (1, 1, seq, seq)
    causal = causal.unsqueeze(0).unsqueeze(0)
    # Padding mask: (batch, 1, 1, seq) - which positions are valid
    padding = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    # Combine: (batch, 1, seq, seq)
    combined = causal * padding
    # Convert to additive mask: 0 -> -inf, 1 -> 0
    return (1.0 - combined) * torch.finfo(dtype).min


class PrefillStageWrapper(torch.nn.Module):
    """Prefill: prompt -> logits + flattened KV."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Create 4D causal mask to bypass transformers' masking_utils (avoids bool & op)
        causal_mask_4d = make_4d_causal_mask(attention_mask, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask_4d,
            use_cache=True,
            return_dict=True,
        )
        kv_flat = flatten_kv_cache(outputs.past_key_values)
        return outputs.logits, *kv_flat


def export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path):
    import iree.turbine.aot as aot

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    wrapper = PrefillStageWrapper(model)
    wrapper.eval()
    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask),
        strict_export=True,
    )

    prefill_mlir_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(prefill_mlir_path))
    print(f"  Saved: {prefill_mlir_path}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return True
