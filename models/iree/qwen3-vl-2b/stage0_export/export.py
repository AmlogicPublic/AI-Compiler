import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT
sys.path.insert(0, str(MODELS_ROOT))

IREE_DIR = Path(__file__).parent
COMPILED_DIR = IREE_DIR / "compiled"

from settings import (
    DECODE_STAGE_NAME,
    EXTERNALIZE_PARAMETERS,
    IMAGE_SIZE,
    IREE_COMPILER_THREADS,
    IREE_LOW_MEMORY_MODE,
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    MODEL_NAME,
    PARAM_ARCHIVE_NAME,
    PARAM_SCOPE,
    PREFILL_STAGE_NAME,
    SAVE_MLIR_BYTECODE,
    STAGE_NAME_LIST,
    TARGET_BACKEND,
    VERIFY_OUTPUT,
    print_export_limitations,
)
from stage_common import (
    compile_stage_to_vmfb,
    create_prefill_example_inputs,
    externalize_model_parameters,
    load_model_and_processor,
    verify_prefill_vmfb,
)
from stage_decode import export_decode_stage_mlir
from stage_prefill import export_prefill_stage_mlir

"""Qwen3-VL-2B -> IREE export entrypoint.
Path: torch.export -> iree-turbine -> IREE vmfb
"""


def main():
    print("=" * 60)
    print("Qwen3-VL-2B -> IREE Export")
    print(f"  target: {TARGET_BACKEND}")
    print("=" * 60)

    print_export_limitations()

    print("\n[1/7] Loading model...")
    model, processor = load_model_and_processor(MODEL_NAME)

    print("\n[2/7] Creating example inputs...")
    prefill_example_inputs = create_prefill_example_inputs(processor, IMAGE_SIZE)

    params_archive = COMPILED_DIR / PARAM_ARCHIVE_NAME
    if EXTERNALIZE_PARAMETERS:
        print("\n[3/7] Externalizing model parameters...")
        assert externalize_model_parameters(model, params_archive, PARAM_SCOPE)
    else:
        params_archive = None

    mlir_suffix = "mlirbc" if SAVE_MLIR_BYTECODE else "mlir"

    print("\n[4/7] Exporting prefill to MLIR...")
    prefill_mlir_path = COMPILED_DIR / f"{PREFILL_STAGE_NAME}.{mlir_suffix}"
    assert export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path)

    print("\n[5/7] Exporting decode to MLIR...")
    decode_mlir_path = COMPILED_DIR / f"{DECODE_STAGE_NAME}.{mlir_suffix}"
    assert export_decode_stage_mlir(
        model,
        decode_mlir_path,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    stage_to_mlir_path = {
        PREFILL_STAGE_NAME: prefill_mlir_path,
        DECODE_STAGE_NAME: decode_mlir_path,
    }
    stage_to_vmfb_path = {
        stage_name: COMPILED_DIR / f"{stage_name}.vmfb"
        for stage_name in STAGE_NAME_LIST
    }
    stage_to_step_title = {
        PREFILL_STAGE_NAME: "\n[6/7] Compiling prefill to IREE vmfb...",
        DECODE_STAGE_NAME: "\n[7/7] Compiling decode to IREE vmfb...",
    }
    for stage_name in STAGE_NAME_LIST:
        print(stage_to_step_title[stage_name])
        assert compile_stage_to_vmfb(
            stage_to_mlir_path[stage_name],
            stage_to_vmfb_path[stage_name],
            target_backend=TARGET_BACKEND,
            low_memory_mode=IREE_LOW_MEMORY_MODE,
            compiler_threads=IREE_COMPILER_THREADS,
        )

    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        assert verify_prefill_vmfb(
            stage_to_vmfb_path[PREFILL_STAGE_NAME],
            prefill_example_inputs,
            params_path=params_archive,
            param_scope=PARAM_SCOPE,
        )

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Prefill MLIR: {stage_to_mlir_path[PREFILL_STAGE_NAME]}")
    print(f"  Prefill VMFB: {stage_to_vmfb_path[PREFILL_STAGE_NAME]}")
    print(f"  Decode MLIR:  {stage_to_mlir_path[DECODE_STAGE_NAME]}")
    print(f"  Decode VMFB:  {stage_to_vmfb_path[DECODE_STAGE_NAME]}")
    if EXTERNALIZE_PARAMETERS:
        print(f"  Parameters:   {params_archive}")
    print("=" * 60)


if __name__ == "__main__":
    main()
