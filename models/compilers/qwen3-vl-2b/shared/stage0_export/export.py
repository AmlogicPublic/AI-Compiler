"""Qwen3-VL shared export entrypoint. Usage: python export.py [iree|tvm]"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="torch._dynamo")
warnings.filterwarnings("ignore", module="torch._export")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
logging.getLogger("torch._export.non_strict_utils").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[5]
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from settings import (
    DECODE_STAGE_NAME,
    IMAGE_SIZE,
    IREE_EXTERNALIZE_PARAMETERS,
    IREE_LOW_MEMORY_MODE,
    IREE_PARAM_ARCHIVE_NAME,
    IREE_PARAM_SCOPE,
    IREE_SAVE_MLIR_BYTECODE,
    IREE_TARGET_BACKEND,
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    MODEL_NAME,
    PREFILL_STAGE_NAME,
    SUPPORTED_BACKENDS,
    TVM_PARAMS_NAME,
    TVM_SAVE_PARAMS_SEPARATELY,
    TVM_TARGET,
    VERIFY_OUTPUT,
    print_export_limitations,
)
from stage_common import (
    compile_stage_to_vmfb,
    compile_to_tvm_lib,
    create_prefill_example_inputs,
    externalize_model_parameters,
    load_model_and_processor,
    save_model_parameters,
    verify_prefill_lib,
    verify_prefill_vmfb,
)
from stage_decode import export_decode_stage_mlir, export_decode_stage_tvm
from stage_prefill import export_prefill_stage_mlir, export_prefill_stage_tvm


MODEL_ROOT = Path(__file__).resolve().parents[2]


def _backend_from_argv() -> str:
    backend = sys.argv[1] if len(sys.argv) > 1 else "iree"
    assert backend in SUPPORTED_BACKENDS, "Usage: python export.py [iree|tvm]"
    return backend


def _export_iree(compiled_dir: Path):
    print("\n[1/7] Loading local checkpoint for export only (no HF runtime eval)...")
    model, processor = load_model_and_processor(MODEL_NAME)

    print("\n[2/7] Creating example inputs...")
    prefill_example_inputs = create_prefill_example_inputs(processor, IMAGE_SIZE)

    params_archive = compiled_dir / IREE_PARAM_ARCHIVE_NAME
    if IREE_EXTERNALIZE_PARAMETERS:
        print("\n[3/7] Externalizing model parameters...")
        assert externalize_model_parameters(model, params_archive, IREE_PARAM_SCOPE)
    else:
        params_archive = None

    mlir_suffix = "mlirbc" if IREE_SAVE_MLIR_BYTECODE else "mlir"

    print("\n[4/7] Exporting prefill to MLIR...")
    prefill_mlir_path = compiled_dir / f"{PREFILL_STAGE_NAME}.{mlir_suffix}"
    assert export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path)

    print("\n[5/7] Exporting decode to MLIR...")
    decode_mlir_path = compiled_dir / f"{DECODE_STAGE_NAME}.{mlir_suffix}"
    assert export_decode_stage_mlir(
        model,
        decode_mlir_path,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    prefill_vmfb_path = compiled_dir / f"{PREFILL_STAGE_NAME}.vmfb"
    decode_vmfb_path = compiled_dir / f"{DECODE_STAGE_NAME}.vmfb"

    print("\n[6/7] Compiling prefill to vmfb...")
    assert compile_stage_to_vmfb(
        prefill_mlir_path,
        prefill_vmfb_path,
        target_backend=IREE_TARGET_BACKEND,
        low_memory_mode=IREE_LOW_MEMORY_MODE,
    )

    print("\n[7/7] Compiling decode to vmfb...")
    assert compile_stage_to_vmfb(
        decode_mlir_path,
        decode_vmfb_path,
        target_backend=IREE_TARGET_BACKEND,
        low_memory_mode=IREE_LOW_MEMORY_MODE,
    )

    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        assert verify_prefill_vmfb(
            prefill_vmfb_path,
            prefill_example_inputs,
            params_path=params_archive,
            param_scope=IREE_PARAM_SCOPE,
        )


def _export_tvm(compiled_dir: Path):
    print("\n[1/7] Loading local checkpoint for export only (no HF runtime eval)...")
    model, processor = load_model_and_processor(MODEL_NAME)

    print("\n[2/7] Creating example inputs...")
    prefill_example_inputs = create_prefill_example_inputs(processor, IMAGE_SIZE)

    print("\n[3/7] Exporting prefill to Relax...")
    prefill_ir_path = compiled_dir / PREFILL_STAGE_NAME
    prefill_mod, prefill_params = export_prefill_stage_tvm(model, prefill_example_inputs, prefill_ir_path)

    print("\n[4/7] Exporting decode to Relax...")
    decode_ir_path = compiled_dir / DECODE_STAGE_NAME
    decode_mod, decode_params = export_decode_stage_tvm(
        model,
        decode_ir_path,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    params_path = compiled_dir / TVM_PARAMS_NAME
    if TVM_SAVE_PARAMS_SEPARATELY:
        print("\n[5/7] Saving model parameters...")
        assert save_model_parameters(model, params_path)

    prefill_lib_path = compiled_dir / f"{PREFILL_STAGE_NAME}.so"
    decode_lib_path = compiled_dir / f"{DECODE_STAGE_NAME}.so"

    print("\n[6/7] Compiling prefill to library...")
    assert compile_to_tvm_lib(prefill_mod, prefill_params, prefill_lib_path, target=TVM_TARGET)

    print("\n[7/7] Compiling decode to library...")
    assert compile_to_tvm_lib(decode_mod, decode_params, decode_lib_path, target=TVM_TARGET)

    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        assert verify_prefill_lib(prefill_lib_path, prefill_example_inputs, params_path=params_path)


def main():
    backend = _backend_from_argv()

    compiled_dir = MODEL_ROOT / f"compiler_{backend}" / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Qwen3-VL-2B -> {backend.upper()} Export")
    print("=" * 60)
    print_export_limitations(backend)

    if backend == "iree":
        _export_iree(compiled_dir)
    else:
        _export_tvm(compiled_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Backend: {backend}")
    print(f"  Compiled dir: {compiled_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
