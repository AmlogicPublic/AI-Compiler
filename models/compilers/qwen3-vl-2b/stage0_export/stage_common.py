"""Shared utilities for Qwen3-VL stage0 export."""

import sys
import time
from pathlib import Path

from PIL import Image
import torch

MODELS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model


def load_model_and_processor(model_name: str):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    local_dir = download_model(model_name)
    print(f"  Checkpoint source: local HF-format files at {local_dir} (export input only)")
    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded for export, params: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def create_prefill_example_inputs(processor, image_size: int):
    image = Image.new("RGB", (image_size, image_size), color=(127, 127, 127))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt", padding=True)


def get_text_model_dims(model):
    config = model.config.text_config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_layers, num_kv_heads, head_dim


def flatten_kv_cache(past_key_values) -> list[torch.Tensor]:
    kv_flat = []
    for keys, values, _ in past_key_values:
        kv_flat.extend([keys, values])
    return kv_flat


def externalize_model_parameters(model, output_path: Path, param_scope: str):
    import iree.turbine.aot as aot

    aot.externalize_module_parameters(model, external_scope=param_scope)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aot.save_module_parameters(str(output_path), model)
    print(f"  Saved parameters: {output_path}")
    return True


def compile_stage_to_vmfb(mlir_path: Path, vmfb_path: Path, *, target_backend: str, low_memory_mode: bool):
    import iree.compiler as ireec

    print(f"  Compiling {mlir_path.name} -> {vmfb_path.name}")
    t0 = time.perf_counter()
    with open(mlir_path, "rb") as f:
        mlir_bytes = f.read()

    extra_args = []
    if target_backend == "llvm-cpu":
        extra_args.append("--iree-llvmcpu-target-cpu=host")
    if low_memory_mode:
        extra_args.append("--mlir-disable-threading")
        extra_args.append("--iree-stream-partitioning-favor=min-peak-memory")

    compiled = ireec.compile_str(
        mlir_bytes,
        target_backends=[target_backend],
        input_type="AUTO",
        extra_args=extra_args,
    )
    vmfb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vmfb_path, "wb") as f:
        f.write(compiled)

    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.1f}s: {vmfb_path}")
    return True


def verify_prefill_vmfb(vmfb_path: Path, prefill_example_inputs, *, params_path: Path | None, param_scope: str):
    import iree.runtime as ireert

    print(f"  Verifying {vmfb_path.name}...")
    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)
    if params_path is not None:
        assert params_path.exists()
        param_index = ireert.ParameterIndex()
        param_index.load(str(params_path))
        param_provider = param_index.create_provider(scope=param_scope)
        param_module = ireert.create_io_parameters_module(ctx.instance, param_provider)
        ctx.add_vm_module(param_module)

    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)
    main_fn = ctx.modules.module["main"]

    outputs = main_fn(
        prefill_example_inputs["input_ids"].numpy(),
        prefill_example_inputs["attention_mask"].numpy(),
        prefill_example_inputs["pixel_values"].numpy(),
    )

    logits = outputs[0].to_host()
    print(f"  Logits shape: {logits.shape}")
    print(f"  KV cache: {len(outputs) - 1} tensors")
    print(f"  Logits sample: {logits[0, -1, :5]}")
    return True
