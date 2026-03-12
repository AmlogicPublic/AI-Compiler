"""Configuration for SmolLM2-135M stage0 export."""

MODEL_NAME = "smollm2-135m"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4
TARGET_BACKEND = "llvm-cpu"  # llvm-cpu | cuda | vulkan
SAVE_MLIR_BYTECODE = True
EXTERNALIZE_PARAMETERS = True
PARAM_ARCHIVE_NAME = "smollm2_135m.irpa"
PARAM_SCOPE = "model"
VERIFY_OUTPUT = True
IREE_LOW_MEMORY_MODE = False  # Small model, no need for low memory mode

PREFILL_STAGE_NAME = "prefill"
DECODE_STAGE_NAME = "decode"

EXPORT_LIMITATIONS = [
    (
        "No streaming generation",
        "Must implement generation loop externally",
        "Exported graph is single forward pass, no control flow",
    ),
    (
        "External parameter archive",
        f"Requires loading {PARAM_ARCHIVE_NAME} at runtime",
        "Model parameters are externalized to keep MLIR compact",
    ),
]


def print_export_limitations() -> None:
    print("\n" + "=" * 60)
    print("EXPORT LIMITATIONS (vs HuggingFace original)")
    print("=" * 60)
    for i, (name, effect, reason) in enumerate(EXPORT_LIMITATIONS, 1):
        print(f"\n{i}. {name}")
        print(f"   Effect: {effect}")
        print(f"   Reason: {reason}")
    print()
