"""Shared utilities for Qwen3-VL stage0 export."""

import sys
import time
from pathlib import Path

import numpy as np
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


def save_model_parameters(model, output_path: Path):
    params = {}
    for name, param in model.named_parameters():
        params[name.replace(".", "_")] = param.detach().cpu().numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **params)
    print(f"  Saved parameters: {output_path}")
    return True


def compile_to_tvm_lib(mod, params: dict, lib_path: Path, *, target: dict):
    import tvm
    from tvm import relax

    print(f"  Compiling to {lib_path.name}...")
    t0 = time.perf_counter()
    target_obj = tvm.target.Target(target)

    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target_obj)

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    ex.export_library(str(lib_path))

    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.1f}s: {lib_path}")
    return True


def load_params_for_tvm(params_path: Path, ir_path: Path, device):
    import re
    import tvm

    assert params_path.exists(), f"Params not found: {params_path}"
    assert ir_path.exists(), f"IR not found: {ir_path}"

    npz_data = np.load(str(params_path))
    with open(ir_path, "r") as f:
        ir_text = f.read()

    param_names = re.compile(r"(p_model_[a-zA-Z0-9_]+):\s*R\.Tensor").findall(ir_text)

    params_tvm = []
    tied_lm_head_key = "model_embed_tokens_weight"
    for pname in param_names:
        npz_key = pname[2:]
        if npz_key not in npz_data and npz_key.startswith("model_model_"):
            npz_key = "model_" + npz_key[len("model_model_"):]
        if npz_key not in npz_data and npz_key == "model_lm_head_weight":
            assert tied_lm_head_key in npz_data
            npz_key = tied_lm_head_key
        assert npz_key in npz_data, f"Param {npz_key} not in npz (from IR: {pname})"
        params_tvm.append(tvm.runtime.tensor(npz_data[npz_key], device=device))

    return params_tvm


def verify_prefill_lib(lib_path: Path, prefill_example_inputs, *, params_path: Path):
    import tvm
    from tvm import relax

    print(f"  Verifying {lib_path.name}...")

    ex = tvm.runtime.load_module(str(lib_path))
    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)

    input_ids_tvm = tvm.runtime.tensor(prefill_example_inputs["input_ids"].numpy(), device=device)
    attention_mask_tvm = tvm.runtime.tensor(prefill_example_inputs["attention_mask"].numpy(), device=device)
    pixel_values_tvm = tvm.runtime.tensor(prefill_example_inputs["pixel_values"].numpy(), device=device)

    params_tvm = load_params_for_tvm(params_path, lib_path.with_suffix(".txt"), device)
    outputs = vm["main"](input_ids_tvm, attention_mask_tvm, pixel_values_tvm, *params_tvm)

    if hasattr(outputs, "numpy"):
        logits = outputs.numpy()
    else:
        assert len(outputs) > 0
        first_output = outputs[0]
        logits = first_output.numpy() if hasattr(first_output, "numpy") else first_output.asnumpy()

    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, -1, :5]}")
    return True
