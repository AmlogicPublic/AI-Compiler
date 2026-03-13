"""Shared configuration for SmolLM2-135M stage0 export."""

MODEL_NAME = "smollm2-135m"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4
PREFILL_SEQ_LEN = 32

PREFILL_STAGE_NAME = "prefill"
DECODE_STAGE_NAME = "decode"

VERIFY_OUTPUT = True

IREE_TARGET_BACKEND = "llvm-cpu"  # llvm-cpu | cuda | vulkan
IREE_SAVE_MLIR_BYTECODE = True
IREE_EXTERNALIZE_PARAMETERS = True
IREE_PARAM_ARCHIVE_NAME = "smollm2_135m.irpa"
IREE_PARAM_SCOPE = "model"
IREE_LOW_MEMORY_MODE = False

TVM_TARGET = {"kind": "llvm", "mtriple": "x86_64-linux-gnu"}
TVM_SAVE_PARAMS_SEPARATELY = True
TVM_PARAMS_NAME = "params.npz"


def print_export_limitations(backend: str) -> None:
    limitations = [
        (
            "No streaming generation",
            "Must implement generation loop externally",
            "Exported graph is single forward pass, no control flow",
        )
    ]
    if backend == "iree":
        limitations.append(
            (
                "External parameter archive",
                f"Requires loading {IREE_PARAM_ARCHIVE_NAME} at runtime",
                "Model parameters are externalized to keep MLIR compact",
            )
        )
    else:
        limitations.append(
            (
                "Separate parameter file",
                f"Requires loading {TVM_PARAMS_NAME} at runtime",
                "Model parameters are saved separately to keep compiled lib small",
            )
        )

    print("\n" + "=" * 60)
    print(f"EXPORT LIMITATIONS [{backend}] (vs HuggingFace original)")
    print("=" * 60)
    for i, (name, effect, reason) in enumerate(limitations, 1):
        print(f"\n{i}. {name}")
        print(f"   Effect: {effect}")
        print(f"   Reason: {reason}")
    print()
