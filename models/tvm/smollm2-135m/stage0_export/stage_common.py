"""Shared utilities for TVM stage0 export."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

MODELS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_dir = download_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def create_prefill_example_inputs(tokenizer, seq_len: int = 32):
    prompt = "The quick brown fox jumps over"
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=seq_len)
    return inputs


def get_text_model_dims(model):
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_layers, num_kv_heads, head_dim


def flatten_kv_cache(past_key_values) -> list[torch.Tensor]:
    kv_flat = []
    for layer_kv in past_key_values:
        kv_flat.extend([layer_kv[0], layer_kv[1]])
    return kv_flat


def save_model_parameters(model, output_path: Path):
    """Save model parameters to npz file."""
    params = {}
    for name, param in model.named_parameters():
        # Replace . with _ for npz key compatibility
        key = name.replace(".", "_")
        params[key] = param.detach().cpu().numpy()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **params)
    print(f"  Saved parameters: {output_path}")
    return True


def compile_to_tvm_lib(
    mod,
    params: dict,
    lib_path: Path,
    *,
    target: str,
):
    """Compile TVM Relax module to shared library."""
    import tvm
    from tvm import relax

    print(f"  Compiling to {lib_path.name}...")
    t0 = time.perf_counter()

    # Build with Relax VM
    if target == "llvm":
        target_obj = tvm.target.Target("llvm")
    elif target == "cuda":
        target_obj = tvm.target.Target("cuda")
    elif target == "vulkan":
        target_obj = tvm.target.Target("vulkan")
    elif target == "metal":
        target_obj = tvm.target.Target("metal")
    else:
        target_obj = tvm.target.Target(target)

    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target_obj)
    
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    ex.export_library(str(lib_path))

    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.1f}s: {lib_path}")
    return True


def verify_prefill_lib(
    lib_path: Path,
    prefill_example_inputs,
    *,
    params_path: Path | None,
):
    """Verify compiled prefill module."""
    import tvm
    from tvm import relax

    print(f"  Verifying {lib_path.name}...")

    ex = tvm.runtime.load_module(str(lib_path))
    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)

    input_ids = prefill_example_inputs["input_ids"].numpy()
    attention_mask = prefill_example_inputs["attention_mask"].numpy()

    # Convert to TVM NDArray
    input_ids_tvm = tvm.nd.array(input_ids, device=device)
    attention_mask_tvm = tvm.nd.array(attention_mask, device=device)

    outputs = vm["main"](input_ids_tvm, attention_mask_tvm)
    
    # First output is logits
    if isinstance(outputs, (list, tuple)):
        logits = outputs[0].numpy()
        print(f"  Logits shape: {logits.shape}")
        print(f"  KV cache: {len(outputs) - 1} tensors")
        print(f"  Logits sample: {logits[0, -1, :5]}")
    else:
        logits = outputs.numpy()
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits sample: {logits[0, -1, :5]}")

    return True
