import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

IREE_DIR = Path(__file__).parent
COMPILED_DIR = IREE_DIR / "compiled"

# ================================================================

from run.shared import download_model
from PIL import Image
import torch
import warnings
import time
from vision_static import StaticQwen3VLVision
from transformers.cache_utils import DynamicCache

"""Qwen3-VL-2B → IREE 导出脚本
路径: torch.export → iree-turbine → IREE vmfb
"""


# ===================== User Macros =====================
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
# =======================================================

# Export limitations vs original HuggingFace model
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


def print_export_limitations():
    """Print export limitations compared to original HuggingFace model"""
    print("\n" + "=" * 60)
    print("EXPORT LIMITATIONS (vs HuggingFace original)")
    print("=" * 60)
    for i, (name, effect, reason) in enumerate(EXPORT_LIMITATIONS, 1):
        print(f"\n{i}. {name}")
        print(f"   Effect: {effect}")
        print(f"   Reason: {reason}")
    print()


class Qwen3VLPrefillWrapper(torch.nn.Module):
    """Prefill: image + prompt → logits + KV cache

    Includes vision encoder. pos_embeds is precomputed to avoid
    data-dependent torch.linspace issue during export.

    Input: input_ids, attention_mask, pixel_values
    Output: logits, key_0, value_0, key_1, value_1, ... (flattened KV cache)
    """

    def __init__(self, model, num_layers, image_grid_thw: torch.Tensor):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        # Register as buffer (constant, not a parameter)
        self.register_buffer("image_grid_thw", image_grid_thw)

        # Replace visual module with a static-shape implementation for export.
        self.model.model.visual = StaticQwen3VLVision(
            model.model.visual, image_grid_thw)
        # Disable deepstack to avoid unsupported bool-index lowering in IREE FX importer.
        self.model.model.visual.deepstack_visual_indexes = []
        self.model.model.visual.deepstack_merger_list = torch.nn.ModuleList()

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=self.image_grid_thw,  # Use hardcoded constant
            use_cache=True,
            return_dict=True,
        )
        # Flatten KV cache: DynamicCache → flat list
        # DynamicCache.__iter__ yields (keys, values, sliding_window_tensor)
        kv_flat = []
        for keys, values, _ in outputs.past_key_values:
            kv_flat.extend([keys, values])
        return outputs.logits, *kv_flat


class Qwen3VLDecodeWrapper(torch.nn.Module):
    """Decode: single token + KV cache → logits + new KV cache

    Input: input_ids (1 token), attention_mask, position_ids, key_0, value_0, ...
    Output: logits, new_key_0, new_value_0, ...
    """

    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, *past_kv_flat):
        # Reconstruct past_key_values: [key_0, value_0, ...] → DynamicCache
        # Use ddp_cache_data to initialize DynamicCache with existing KV tensors
        cache_data = [
            (past_kv_flat[i * 2], past_kv_flat[i * 2 + 1])
            for i in range(self.num_layers)
        ]
        cache = DynamicCache(ddp_cache_data=cache_data)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        # Flatten new KV cache: DynamicCache → flat list
        new_kv_flat = []
        for keys, values, _ in outputs.past_key_values:
            new_kv_flat.extend([keys, values])
        return outputs.logits, *new_kv_flat


def load_model():
    """Load Qwen3-VL model and processor"""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    local_dir = download_model(MODEL_NAME)

    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        local_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )

    model.eval()
    print(
        f"Model loaded, params: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def create_example_inputs(processor):
    """Create example inputs for tracing"""
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(127, 127, 127))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt", padding=True)
    return inputs


def get_model_config(model):
    """Extract model config for KV cache shapes"""
    config = model.config.text_config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads",
                           config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    return num_layers, num_kv_heads, head_dim


def externalize_and_save_parameters(model, output_path: Path):
    """Externalize model parameters to keep MLIR small."""
    import iree.turbine.aot as aot

    aot.externalize_module_parameters(model, external_scope=PARAM_SCOPE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aot.save_module_parameters(str(output_path), model)
    print(f"  Saved parameters: {output_path}")
    return True


def export_prefill_mlir(model, example_inputs, output_path: Path):
    """Export prefill: image + prompt → logits + KV cache

    Includes vision encoder. image_grid_thw is hardcoded as constant.
    """
    import os
    import iree.turbine.aot as aot

    # Disable torch_compilable_check in transformers - IREE FX importer
    # doesn't support torch._check_with which uses builtin ge/le operators
    os.environ["TRANSFORMERS_DISABLE_TORCH_CHECK"] = "1"

    num_layers, num_kv_heads, head_dim = get_model_config(model)
    print(
        f"  Model config: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    input_ids = example_inputs["input_ids"]
    attention_mask = example_inputs["attention_mask"]
    pixel_values = example_inputs["pixel_values"]
    image_grid_thw = example_inputs["image_grid_thw"]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  image_grid_thw: {image_grid_thw} (hardcoded as constant)")

    # Hardcode image_grid_thw into wrapper to avoid data-dependent linspace
    wrapper = Qwen3VLPrefillWrapper(model, num_layers, image_grid_thw)
    wrapper.eval()

    # strict_export=True: let TorchDynamo handle Python control flow
    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask, pixel_values),
        strict_export=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(output_path))
    print(f"  Saved: {output_path}")
    print(f"  Output: logits + {num_layers * 2} KV tensors")
    return True


def export_decode_mlir(model, output_path: Path):
    """Export decode: 1 token + KV cache → logits + new KV cache"""
    import iree.turbine.aot as aot
    from torch.export import Dim

    num_layers, num_kv_heads, head_dim = get_model_config(model)

    wrapper = Qwen3VLDecodeWrapper(model, num_layers)
    wrapper.eval()

    # Example inputs for decode step
    batch_size = 1
    seq_len = 100  # Example cached sequence length

    input_ids = torch.randint(0, 1000, (batch_size, 1))  # 1 new token
    attention_mask = torch.ones(batch_size, seq_len + 1, dtype=torch.long)
    position_ids = torch.tensor([[seq_len]], dtype=torch.long)

    # KV cache: [batch, num_kv_heads, seq_len, head_dim]
    past_kv_flat = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        past_kv_flat.extend([k, v])

    print(f"  input_ids: {input_ids.shape} (1 new token)")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  position_ids: {position_ids.shape}")
    print(
        f"  KV cache: {num_layers} layers × 2 × [{batch_size}, {num_kv_heads}, {seq_len}, {head_dim}]")

    # Use AUTO hints so export can fall back to static specialization when needed.
    batch_dim = Dim.AUTO(min=1, max=MAX_BATCH_SIZE)
    seq_dim = Dim.AUTO(min=1, max=MAX_SEQ_LEN)
    kv_seq_dim = Dim.AUTO(min=1, max=MAX_SEQ_LEN - 1)

    kv_dynamic_shapes = []
    # Add dynamic shapes for each KV tensor
    for i in range(num_layers * 2):
        # seq_dim - 1 because KV cache is previous tokens
        kv_dynamic_shapes.append(
            {0: batch_dim, 2: kv_seq_dim}
        )

    # Use tuple/list form because *past_kv_flat is a vararg in forward signature.
    dynamic_shapes = (
        {0: batch_dim},
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim},
        tuple(kv_dynamic_shapes),
    )

    # Note: strict=True works for decode (no vision encoder)
    exported = aot.export(
        wrapper,
        args=(input_ids, attention_mask, position_ids, *past_kv_flat),
        strict_export=True,
        dynamic_shapes=dynamic_shapes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported.save_mlir(str(output_path))
    print(f"  Saved: {output_path}")
    print(f"  Dynamic: batch=[1,{MAX_BATCH_SIZE}], seq=[1,{MAX_SEQ_LEN}]")
    return True


def compile_to_vmfb(mlir_path: Path, vmfb_path: Path, target="llvm-cpu"):
    """Compile MLIR/MLIRBC to IREE vmfb"""
    try:
        import iree.compiler as ireec
    except ImportError:
        print("ERROR: iree-compiler not installed.")
        print("Install with: pip install iree-compiler")
        return False

    print(f"  Compiling {mlir_path.name} → {vmfb_path.name}")
    t0 = time.perf_counter()

    with open(mlir_path, "rb") as f:
        mlir_bytes = f.read()

    compiled = ireec.compile_str(
        mlir_bytes,
        target_backends=[target],
        input_type="AUTO",
    )

    vmfb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vmfb_path, "wb") as f:
        f.write(compiled)

    elapsed = time.perf_counter() - t0
    print(f"  Compiled in {elapsed:.1f}s: {vmfb_path}")
    return True


def verify_vmfb(vmfb_path: Path, example_inputs, params_path: Path | None = None):
    """Verify compiled vmfb produces output"""
    import iree.runtime as ireert

    print(f"  Verifying {vmfb_path.name}...")

    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)
    if params_path is not None:
        assert params_path.exists()
        param_index = ireert.ParameterIndex()
        param_index.load(str(params_path))
        param_provider = param_index.create_provider(scope=PARAM_SCOPE)
        param_module = ireert.create_io_parameters_module(
            ctx.instance, param_provider)
        ctx.add_vm_module(param_module)

    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)

    main_fn = ctx.modules.module["main"]

    # Convert inputs to numpy
    input_ids = example_inputs["input_ids"].numpy()
    attention_mask = example_inputs["attention_mask"].numpy()
    pixel_values = example_inputs["pixel_values"].numpy()
    # Run inference - returns (logits, k0, v0, k1, v1, ...)
    outputs = main_fn(input_ids, attention_mask, pixel_values)

    # First output is logits
    if hasattr(outputs, "__len__") and len(outputs) > 1:
        logits = outputs[0].to_host()
        num_kv = len(outputs) - 1
        print(f"  Logits shape: {logits.shape}")
        print(f"  KV cache: {num_kv} tensors")
    else:
        logits = outputs.to_host()
        print(f"  Logits shape: {logits.shape}")

    print(f"  Logits sample: {logits[0, -1, :5]}")
    return True


def main():
    print("=" * 60)
    print("Qwen3-VL-2B → IREE Export")
    print(f"  target: {TARGET_BACKEND}")
    print("=" * 60)

    print_export_limitations()

    # Load model
    print("\n[1/7] Loading model...")
    model, processor = load_model()

    # Create example inputs
    print("\n[2/7] Creating example inputs...")
    example_inputs = create_example_inputs(processor)

    params_archive = COMPILED_DIR / PARAM_ARCHIVE_NAME
    if EXTERNALIZE_PARAMETERS:
        print("\n[3/7] Externalizing model parameters...")
        assert externalize_and_save_parameters(model, params_archive)
    else:
        params_archive = None

    mlir_suffix = "mlirbc" if SAVE_MLIR_BYTECODE else "mlir"

    # Export prefill to MLIR
    print("\n[4/7] Exporting prefill to MLIR...")
    prefill_mlir = COMPILED_DIR / f"prefill.{mlir_suffix}"
    assert export_prefill_mlir(model, example_inputs, prefill_mlir)

    # Export decode to MLIR
    print("\n[5/7] Exporting decode to MLIR...")
    decode_mlir = COMPILED_DIR / f"decode.{mlir_suffix}"
    assert export_decode_mlir(model, decode_mlir)

    # Compile to vmfb
    print("\n[6/7] Compiling prefill to IREE vmfb...")
    prefill_vmfb = COMPILED_DIR / "prefill.vmfb"
    assert compile_to_vmfb(prefill_mlir, prefill_vmfb, target=TARGET_BACKEND)

    print("\n[7/7] Compiling decode to IREE vmfb...")
    decode_vmfb = COMPILED_DIR / "decode.vmfb"
    assert compile_to_vmfb(decode_mlir, decode_vmfb, target=TARGET_BACKEND)

    # Verify
    if VERIFY_OUTPUT:
        print("\n[Verify] Running prefill verification...")
        verify_vmfb(prefill_vmfb, example_inputs, params_archive)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Prefill MLIR: {prefill_mlir}")
    print(f"  Prefill VMFB: {prefill_vmfb}")
    print(f"  Decode MLIR:  {decode_mlir}")
    print(f"  Decode VMFB:  {decode_vmfb}")
    if EXTERNALIZE_PARAMETERS:
        print(f"  Parameters:   {params_archive}")
    print("=" * 60)


if __name__ == "__main__":
    main()
