"""Shared configuration for Qwen3-VL-2B stage0 export."""

MODEL_NAME = "qwen3-vl-2b"
IMAGE_SIZE = 448
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 4

PREFILL_STAGE_NAME = "prefill"
DECODE_STAGE_NAME = "decode"

SUPPORTED_BACKENDS = {"iree"}
VERIFY_OUTPUT = True

IREE_TARGET_BACKEND = "llvm-cpu"  # llvm-cpu | cuda | vulkan
IREE_SAVE_MLIR_BYTECODE = True
IREE_EXTERNALIZE_PARAMETERS = True
IREE_PARAM_ARCHIVE_NAME = "qwen3_vl_2b.irpa"
IREE_PARAM_SCOPE = "model"
IREE_LOW_MEMORY_MODE = True


def print_export_limitations(backend: str) -> None:
    limitations = [
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
    ]

    if backend == "iree":
        limitations.append(
            (
                "External parameter archive",
                f"Requires loading {IREE_PARAM_ARCHIVE_NAME} at runtime",
                "Model parameters are externalized to keep MLIR compact",
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
