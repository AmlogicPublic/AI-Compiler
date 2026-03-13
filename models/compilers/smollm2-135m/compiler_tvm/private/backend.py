import importlib.util
from pathlib import Path

import numpy as np
import tvm
from tvm import relax


MODEL_ROOT = Path(__file__).resolve().parents[2]
COMPILED_DIR = MODEL_ROOT / "compiler_tvm" / "compiled"
STAGE0_EXPORT_DIR = MODEL_ROOT / "stage0_export"


def _load_stage_common_module():
    stage_common_path = STAGE0_EXPORT_DIR / "stage_common.py"
    assert stage_common_path.exists(), f"Missing stage_common.py: {stage_common_path}"
    spec = importlib.util.spec_from_file_location("smollm2_stage_common", stage_common_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tvm_module(lib_path: Path, device):
    assert lib_path.exists(), f"Library not found: {lib_path}\nRun 'python stage0_export/export.py tvm' first"
    ex = tvm.runtime.load_module(str(lib_path))
    return relax.VirtualMachine(ex, device)


def load_params_for_module(lib_path: Path, device) -> list:
    params_path = COMPILED_DIR / "params.npz"
    ir_path = lib_path.with_suffix(".txt")
    assert params_path.exists(), f"Params not found: {params_path}\nRun 'python stage0_export/export.py tvm' first"
    assert ir_path.exists(), f"IR not found: {ir_path}\nRun 'python stage0_export/export.py tvm' first"
    return _load_stage_common_module().load_params_for_tvm(params_path, ir_path, device)


def unpack_tvm_outputs(outputs):
    if hasattr(outputs, "numpy"):
        return outputs.numpy(), []

    assert len(outputs) > 0
    first = outputs[0]
    logits = first.numpy() if hasattr(first, "numpy") else first.asnumpy()
    kv_cache = [outputs[i] for i in range(1, len(outputs))]
    return logits, kv_cache


def to_tvm_decode_inputs(input_ids, position_ids, cache_position, kv_cache, device):
    input_ids_tvm = tvm.runtime.tensor(input_ids, device=device)
    position_ids_tvm = tvm.runtime.tensor(position_ids, device=device)
    cache_position_tvm = tvm.runtime.tensor(cache_position, device=device)
    kv_cache_tvm = [tvm.runtime.tensor(kv, device=device) if isinstance(kv, np.ndarray) else kv for kv in kv_cache]
    return input_ids_tvm, position_ids_tvm, cache_position_tvm, kv_cache_tvm
