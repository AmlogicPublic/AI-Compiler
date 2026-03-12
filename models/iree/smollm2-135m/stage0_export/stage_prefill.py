"""Prefill stage export."""

import torch

from stage_common import flatten_kv_cache, get_text_model_dims


class PrefillStageWrapper(torch.nn.Module):
    """Prefill: prompt -> logits + flattened KV."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
