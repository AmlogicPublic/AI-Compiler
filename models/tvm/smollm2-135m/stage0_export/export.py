"""SmolLM2-135M -> TVM export entrypoint.
Path: torch.export -> TVM Relax -> compiled library
"""

import logging
import sys
import warnings
from pathlib import Path

# Suppress third-party library warnings
warnings.filterwarnings("ignore", module="torch._dynamo")
warnings.filterwarnings("ignore", module="torch._export")
warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")

logging.getLogger("torch._export.non_strict_utils").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT
sys.path.insert(0, str(MODELS_ROOT))

TVM_DIR = Path(__file__).parent
COMPILED_DIR = TVM_DIR / "compiled"

from settings import *
from stage_common import (
    compile_to_tvm_lib,
    create_prefill_example_inputs,
    load_model_and_tokenizer,
    save_model_parameters,
    verify_prefill_lib,
)
from stage_decode import export_decode_stage_tvm
from stage_prefill import export_prefill_stage_tvm


def main():
    print("=" * 60)
    print("SmolLM2-135M -> TVM Export")
    print(f"  target: {TARGET}")
    print("=" * 60)

    print_export_limitations()

    print("\n[1/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    print("\n[2/6] Creating example inputs...")
    prefill_example_inputs = create_prefill_example_inputs(tokenizer)

    params_path = COMPILED_DIR / "params.npz"
    if SAVE_PARAMS_SEPARATELY:
        print("\n[3/6] Saving model parameters...")
        assert save_model_parameters(model, params_path)
    else:
        params_path = None

    print("\n[4/6] Exporting prefill to TVM Relax...")
    prefill_ir_path = COMPILED_DIR / PREFILL_STAGE_NAME
    prefill_mod, prefill_params = export_prefill_stage_tvm(
        model, prefill_example_inputs, prefill_ir_path
    )

    print("\n[5/6] Exporting decode to TVM Relax...")
    decode_ir_path = COMPILED_DIR / DECODE_STAGE_NAME
    decode_mod, decode_params = export_decode_stage_tvm(
        model,
        decode_ir_path,
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )

    prefill_lib_path = COMPILED_DIR / f"{PREFILL_STAGE_NAME}.so"
    decode_lib_path = COMPILED_DIR / f"{DECODE_STAGE_NAME}.so"

    print("\n[6/6] Compiling prefill to TVM library...")
    assert compile_to_tvm_lib(
        prefill_mod,
        prefill_params,
        prefill_lib_path,
        target=TARGET,
    )

    print("\n[6/6] Compiling decode to TVM library...")
    assert compile_to_tvm_lib(
        decode_mod,
        decode_params,
        decode_lib_path,
        target=TARGET,
    )

    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        assert verify_prefill_lib(
            prefill_lib_path,
            prefill_example_inputs,
            params_path=params_path,
        )

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Prefill IR:  {prefill_ir_path}.json")
    print(f"  Prefill Lib: {prefill_lib_path}")
    print(f"  Decode IR:   {decode_ir_path}.json")
    print(f"  Decode Lib:  {decode_lib_path}")
    if SAVE_PARAMS_SEPARATELY:
        print(f"  Parameters:  {params_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
