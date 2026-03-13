"""Shared utilities for TVM stage0 export."""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import tvm

MODELS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model


def patch_dynamic_cache_for_tvm_export() -> None:
    """
    Patch transformers DynamicLayer update path for TVM Relax translation.

    The default first update goes through an empty-cache concat path and can
    materialize zero-length intermediate tensors in exported IR. This triggers
    Relax compile issues in some passes. We keep the same KV semantics by making
    the first update directly assign key/value tensors.
    """
    from transformers.cache_utils import DynamicLayer

    if getattr(DynamicLayer, "_tvm_export_patched", False):
        return

    def _update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
            return self.keys, self.values

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    DynamicLayer.update = _update
    DynamicLayer._tvm_export_patched = True


def load_model_and_tokenizer(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    patch_dynamic_cache_for_tvm_export()

    local_dir = download_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        trust_remote_code=True,
        dtype=torch.float32,
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
    target: dict,
):
    """Compile TVM Relax module to shared library."""
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
    """Load params from npz and order them according to IR function signature."""
    import re

    assert params_path.exists(), f"Params not found: {params_path}"
    assert ir_path.exists(), f"IR not found: {ir_path}"

    # Load npz
    npz_data = np.load(str(params_path))

    # Parse IR to get param names in order (p_model_xxx format)
    with open(ir_path, "r") as f:
        ir_text = f.read()

    # Extract param names from function signature (skip first 2-64 inputs)
    # Pattern: p_model_xxx: R.Tensor(...)
    param_pattern = re.compile(r"(p_model_[a-zA-Z0-9_]+):\s*R\.Tensor")
    param_names = param_pattern.findall(ir_text)

    params_tvm = []
    tied_lm_head_key = "model_embed_tokens_weight"
    for pname in param_names:
        # Convert p_model_xxx to model_xxx (remove p_ prefix)
        npz_key = pname[2:]  # remove "p_"
        if npz_key not in npz_data and npz_key.startswith("model_model_"):
            npz_key = "model_" + npz_key[len("model_model_") :]
        if npz_key not in npz_data and npz_key == "model_lm_head_weight":
            assert tied_lm_head_key in npz_data, (
                f"Expected tied lm_head alias {tied_lm_head_key} in npz "
                f"(from IR: {pname})"
            )
            npz_key = tied_lm_head_key
        assert npz_key in npz_data, f"Param {npz_key} not in npz (from IR: {pname})"
        arr = npz_data[npz_key]
        params_tvm.append(tvm.runtime.tensor(arr, device=device))

    return params_tvm


def verify_prefill_lib(
    lib_path: Path,
    prefill_example_inputs,
    *,
    params_path: Path | None,
):
    """Verify compiled prefill module."""
    from tvm import relax

    print(f"  Verifying {lib_path.name}...")

    ex = tvm.runtime.load_module(str(lib_path))
    device = tvm.cpu()
    vm = relax.VirtualMachine(ex, device)

    input_ids = prefill_example_inputs["input_ids"].numpy()
    attention_mask = prefill_example_inputs["attention_mask"].numpy()

    input_ids_tvm = tvm.runtime.tensor(input_ids, device=device)
    attention_mask_tvm = tvm.runtime.tensor(attention_mask, device=device)

    # Load params from npz
    ir_path = lib_path.with_suffix(".txt")
    params_tvm = load_params_for_tvm(params_path, ir_path, device)

    outputs = vm["main"](input_ids_tvm, attention_mask_tvm, *params_tvm)

    if hasattr(outputs, "numpy"):
        logits = outputs.numpy()
    else:
        assert len(outputs) > 0, "VM main output is empty"
        first_output = outputs[0]
        if hasattr(first_output, "numpy"):
            logits = first_output.numpy()
        else:
            logits = first_output.asnumpy()

    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits sample: {logits[0, -1, :5]}")

    return True
