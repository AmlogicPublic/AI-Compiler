"""Shared prefill stage export for SmolLM2."""

import torch

from stage_common import flatten_kv_cache, get_text_model_dims


def make_4d_causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch_size, seq_len = attention_mask.shape
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=attention_mask.device))
    causal = causal.unsqueeze(0).unsqueeze(0)
    padding = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
    combined = causal * padding
    return (1.0 - combined) * torch.finfo(dtype).min


class PrefillStageWrapper(torch.nn.Module):
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
        return outputs.logits, *flatten_kv_cache(outputs.past_key_values)


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
    exported = aot.export(wrapper, args=(input_ids, attention_mask), strict_export=True)

    prefill_mlir_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(prefill_mlir_path))
    print(f"  Saved: {prefill_mlir_path}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return True


def export_prefill_stage_tvm(model, prefill_example_inputs, output_path):
    from tvm.relax.frontend.torch import from_exported_program  # type: ignore

    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")

    wrapper = PrefillStageWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        exported = torch.export.export(wrapper, args=(input_ids, attention_mask), strict=False)

    mod = from_exported_program(exported, keep_params_as_input=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(str(mod))
    print(f"  Saved IR: {output_path.with_suffix('.txt')}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return mod, {}
