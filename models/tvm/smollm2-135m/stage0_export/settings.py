"""Configuration for SmolLM2-135M TVM export."""

MODEL_NAME = "smollm2-135m"
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4
TARGET = "llvm"  # llvm | cuda | vulkan | metal
SAVE_PARAMS_SEPARATELY = True
VERIFY_OUTPUT = True

PREFILL_STAGE_NAME = "prefill"
DECODE_STAGE_NAME = "decode"

EXPORT_LIMITATIONS = [
    (
        "No streaming generation",
        "Must implement generation loop externally",
        "Exported graph is single forward pass, no control flow",
    ),
    (
        "Separate parameter file",
        "Requires loading params.npz at runtime",
        "Model parameters are saved separately to keep compiled lib small",
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
