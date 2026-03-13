from pathlib import Path

import numpy as np
import tvm
from tvm import relax

from shared.stage0_export.stage_common import load_params_for_tvm
from private import MODEL_ROOT

COMPILED_DIR = MODEL_ROOT / "compiler_tvm" / "compiled"


def load_tvm_module(lib_path: Path, device):
    assert lib_path.exists(), f"Library not found: {lib_path}\nRun 'python shared/stage0_export/export.py tvm' first"
    ex = tvm.runtime.load_module(str(lib_path))
    return relax.VirtualMachine(ex, device)


def load_params_for_module(lib_path: Path, device) -> list:
    params_path = COMPILED_DIR / "params.npz"
    ir_path = lib_path.with_suffix(".txt")
    assert params_path.exists(), f"Params not found: {params_path}\nRun 'python shared/stage0_export/export.py tvm' first"
    assert ir_path.exists(), f"IR not found: {ir_path}\nRun 'python shared/stage0_export/export.py tvm' first"
    return load_params_for_tvm(params_path, ir_path, device)


def unpack_tvm_outputs(outputs):
    if hasattr(outputs, "numpy"):
        return outputs.numpy(), []

    assert len(outputs) > 0
    first = outputs[0]
    logits = first.numpy() if hasattr(first, "numpy") else first.asnumpy()
    kv_cache = [outputs[i] for i in range(1, len(outputs))]
    return logits, kv_cache


def to_tvm_prefill_inputs(inputs: dict, device):
    input_ids_tvm = tvm.runtime.tensor(inputs["input_ids"], device=device)
    attention_mask_tvm = tvm.runtime.tensor(inputs["attention_mask"], device=device)
    pixel_values_tvm = tvm.runtime.tensor(inputs["pixel_values"], device=device)
    return input_ids_tvm, attention_mask_tvm, pixel_values_tvm


def to_tvm_decode_inputs(input_ids, attention_mask, position_ids, kv_cache, device):
    input_ids_tvm = tvm.runtime.tensor(input_ids, device=device)
    attention_mask_tvm = tvm.runtime.tensor(attention_mask, device=device)
    position_ids_tvm = tvm.runtime.tensor(position_ids, device=device)
    kv_cache_tvm = [tvm.runtime.tensor(kv, device=device) if isinstance(kv, np.ndarray) else kv for kv in kv_cache]
    return input_ids_tvm, attention_mask_tvm, position_ids_tvm, kv_cache_tvm
