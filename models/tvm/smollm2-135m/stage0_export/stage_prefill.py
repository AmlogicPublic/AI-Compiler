"""Prefill stage export for TVM."""

import torch
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

from stage_common import flatten_kv_cache, get_text_model_dims


def make_4d_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a 4D causal attention mask from a 2D padding mask.
    Returns: (batch, 1, seq_len, seq_len) float mask where 0.0=attend, -inf=ignore
    """
    batch_size, seq_len = attention_mask.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=attention_mask.device))
    causal = causal.unsqueeze(0).unsqueeze(0)
    padding = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    combined = causal * padding
    return (1.0 - combined) * torch.finfo(dtype).min


class PrefillStageWrapper(torch.nn.Module):
    """Prefill: prompt -> logits + flattened KV."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        causal_mask_4d = make_4d_causal_mask(attention_mask, self.model.dtype)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=causal_mask_4d,
            use_cache=True,
            return_dict=True,
        )
        kv_flat = flatten_kv_cache(outputs.past_key_values)
        return outputs.logits, *kv_flat


def export_prefill_stage_tvm(model, prefill_example_inputs, output_path):
    """Export prefill stage to TVM Relax IR."""
    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    wrapper = PrefillStageWrapper(model)
    wrapper.eval()

    # Export via torch.export
    exported = torch.export.export(
        wrapper,
        args=(input_ids, attention_mask),
        strict=True,
    )

    # Convert to TVM Relax
    mod = from_exported_program(exported, keep_params_as_input=True)

    # Extract params from exported program
    params = {}
    state_dict = dict(exported.state_dict)
    for name, tensor in state_dict.items():
        params[name] = tensor.detach().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save Relax IR
    ir_path = output_path.with_suffix(".json")
    with open(ir_path, "w") as f:
        f.write(mod.as_text())
    print(f"  Saved IR: {ir_path}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")

    return mod, params
