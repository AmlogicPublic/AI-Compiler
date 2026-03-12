"""Configuration for Qwen3-VL stage0 export."""

MODEL_NAME = "qwen3-vl-2b"
IMAGE_SIZE = 448
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4
TARGET_BACKEND = "llvm-cpu"  # llvm-cpu | cuda | vulkan
SAVE_MLIR_BYTECODE = True
EXTERNALIZE_PARAMETERS = True
PARAM_ARCHIVE_NAME = "qwen3_vl_2b.irpa"
PARAM_SCOPE = "model"
VERIFY_OUTPUT = True
IREE_LOW_MEMORY_MODE = True
IREE_COMPILER_THREADS = 1

PREFILL_STAGE_NAME = "prefill"
DECODE_STAGE_NAME = "decode"
STAGE_NAME_LIST = [PREFILL_STAGE_NAME, DECODE_STAGE_NAME]

EXPORT_LIMITATIONS = [
    (
        "Fixed image size",
        f"Only {IMAGE_SIZE}x{IMAGE_SIZE} images supported",
        "image_grid_thw hardcoded as constant to avoid data-dependent torch.linspace",
    ),
    (
        "No streaming generation",
        "Must implement generation loop externally",
        "Exported graph is single forward pass, no control flow",
    ),
    (
        "Deepstack visual fusion disabled",
        "Slightly lower multimodal fusion quality in early text layers",
        "Avoids bool-index path in transformers _deepstack_process that exports unsupported builtin ge",
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
