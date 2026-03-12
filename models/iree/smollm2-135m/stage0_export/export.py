import logging
import sys
import warnings
from pathlib import Path

# Suppress third-party library warnings
warnings.filterwarnings("ignore", module="torch._dynamo")
warnings.filterwarnings("ignore", module="torch._export")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")

# Suppress torch logging warnings (Dim.AUTO specialization)
logging.getLogger("torch._export.non_strict_utils").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT
sys.path.insert(0, str(MODELS_ROOT))

IREE_DIR = Path(__file__).parent
COMPILED_DIR = IREE_DIR / "compiled"

from settings import *
from stage_common import *

from stage_decode import export_decode_stage_mlir
from stage_prefill import export_prefill_stage_mlir

"""SmolLM2-135M -> IREE export entrypoint.
Path: torch.export -> iree-turbine -> IREE vmfb
"""


def main():
    print("=" * 60)
    print("SmolLM2-135M -> IREE Export")
    print(f"  target: {TARGET_BACKEND}")
    print("=" * 60)

    print_export_limitations()

    print("\n[1/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("\n[2/6] Creating example inputs...")
    prefill_example_inputs = create_prefill_example_inputs(tokenizer)

    params_archive = COMPILED_DIR / PARAM_ARCHIVE_NAME
    if EXTERNALIZE_PARAMETERS:
        print("\n[3/6] Externalizing model parameters...")
        assert externalize_model_parameters(model, params_archive, PARAM_SCOPE)
    else:
        params_archive = None

    mlir_suffix = "mlirbc" if SAVE_MLIR_BYTECODE else "mlir"

    print("\n[4/6] Exporting prefill to MLIR...")
    prefill_mlir_path = COMPILED_DIR / f"{PREFILL_STAGE_NAME}.{mlir_suffix}"
    assert export_prefill_stage_mlir(model, prefill_example_inputs, prefill_mlir_path)

    print("\n[5/6] Exporting decode to MLIR...")
    decode_mlir_path = COMPILED_DIR / f"{DECODE_STAGE_NAME}.{mlir_suffix}"
    assert export_decode_stage_mlir(
        model,
        decode_mlir_path,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    prefill_vmfb_path = COMPILED_DIR / f"{PREFILL_STAGE_NAME}.vmfb"
    decode_vmfb_path = COMPILED_DIR / f"{DECODE_STAGE_NAME}.vmfb"

    print("\n[6/6] Compiling prefill to IREE vmfb...")
    assert compile_stage_to_vmfb(
        prefill_mlir_path,
        prefill_vmfb_path,
        target_backend=TARGET_BACKEND,
        low_memory_mode=IREE_LOW_MEMORY_MODE,
    )

    print("\n[6/6] Compiling decode to IREE vmfb...")
    assert compile_stage_to_vmfb(
        decode_mlir_path,
        decode_vmfb_path,
        target_backend=TARGET_BACKEND,
        low_memory_mode=IREE_LOW_MEMORY_MODE,
    )

    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        assert verify_prefill_vmfb(
            prefill_vmfb_path,
            prefill_example_inputs,
            params_path=params_archive,
            param_scope=PARAM_SCOPE,
        )

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Prefill MLIR: {prefill_mlir_path}")
    print(f"  Prefill VMFB: {prefill_vmfb_path}")
    print(f"  Decode MLIR:  {decode_mlir_path}")
    print(f"  Decode VMFB:  {decode_vmfb_path}")
    if EXTERNALIZE_PARAMETERS:
        print(f"  Parameters:   {params_archive}")
    print("=" * 60)


if __name__ == "__main__":
    main()
