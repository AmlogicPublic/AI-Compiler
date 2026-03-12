"""Prefill stage export."""

import os

import torch

from stage_common import get_text_model_dims
from vision_static import StaticQwen3VLVision


def flatten_kv_cache(past_key_values) -> list[torch.Tensor]:
    kv_flat = []
    for keys, values, _ in past_key_values:
        kv_flat.extend([keys, values])
    return kv_flat


def adapt_visual_for_prefill_export(model, image_grid_thw: torch.Tensor) -> None:
    model.model.visual = StaticQwen3VLVision(model.model.visual, image_grid_thw)
    model.model.visual.deepstack_visual_indexes = []
    model.model.visual.deepstack_merger_list = torch.nn.ModuleList()


class PrefillStageWrapper(torch.nn.Module):
    """Prefill: image + prompt -> logits + flattened KV."""

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
        kv_flat = flatten_kv_cache(outputs.past_key_values)
        return outputs.logits, *kv_flat


def export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path):
    import iree.turbine.aot as aot

    os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"
    num_layers, num_kv_heads, head_dim = get_text_model_dims(model)
    print(f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = prefill_example_inputs["input_ids"]
    attention_mask = prefill_example_inputs["attention_mask"]
    pixel_values = prefill_example_inputs["pixel_values"]
    image_grid_thw = prefill_example_inputs["image_grid_thw"]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  image_grid_thw: {image_grid_thw} (hardcoded as constant)")

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
