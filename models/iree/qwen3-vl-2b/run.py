"""Qwen3-VL-2B IREE Runtime 推理
使用编译好的 vmfb 文件运行推理
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from run.shared import download_model
from PIL import Image
import numpy as np
import time
import os


IREE_DIR = Path(__file__).parent
COMPILED_DIR = IREE_DIR / "compiled"

# ===================== User Macros =====================
MODEL_NAME = "qwen3-vl-2b"
IMAGE_PATH = ""  # empty = use demo image
QUESTION = "Describe this image."
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
# =======================================================


def load_processor():
    """Load Qwen3-VL processor (for tokenization)"""
    from transformers import AutoProcessor

    local_dir = download_model(MODEL_NAME)

    processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)

    return processor


def load_iree_module(vmfb_path: Path):
    """Load compiled IREE module"""
    import iree.runtime as ireert

    assert vmfb_path.exists(
    ), f"VMFB not found: {vmfb_path}\nRun 'python export.py' first"

    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)

    return ctx


class Qwen3VLIREERunner:
    """IREE-based Qwen3-VL runner with KV Cache support"""

    def __init__(self):
        print("Loading IREE modules...")

        self.processor = load_processor()
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load prefill module (image + prompt → logits + KV cache)
        prefill_vmfb = COMPILED_DIR / "prefill.vmfb"
        assert prefill_vmfb.exists(
        ), f"Run 'python export.py' first: {prefill_vmfb}"
        self.prefill_ctx = load_iree_module(prefill_vmfb)
        self.prefill_fn = self.prefill_ctx.modules.module["main"]

        # Load decode module (1 token + KV cache → logits + new KV cache)
        decode_vmfb = COMPILED_DIR / "decode.vmfb"
        if decode_vmfb.exists():
            self.decode_ctx = load_iree_module(decode_vmfb)
            self.decode_fn = self.decode_ctx.modules.module["main"]
        else:
            print(f"  Warning: {decode_vmfb} not found, decode disabled")
            self.decode_fn = None

        print("IREE runner ready")

    def prepare_inputs(self, image: Image.Image, question: str):
        """Prepare inputs for IREE inference"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True)

        return {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
            "pixel_values": inputs["pixel_values"].numpy(),
            "image_grid_thw": inputs["image_grid_thw"].numpy(),
        }

    def forward_prefill(self, inputs: dict):
        """Prefill: image + prompt → logits + KV cache"""
        outputs = self.prefill_fn(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        # outputs: (logits, k0, v0, k1, v1, ...)
        logits = outputs[0].to_host()
        kv_cache = [outputs[i].to_host() for i in range(1, len(outputs))]
        return logits, kv_cache

    def forward_decode(self, input_ids, attention_mask, position_ids, kv_cache):
        """Decode: 1 token + KV cache → logits + new KV cache"""
        assert self.decode_fn is not None, "Decode module not loaded"
        outputs = self.decode_fn(
            input_ids, attention_mask, position_ids, *kv_cache)
        logits = outputs[0].to_host()
        new_kv_cache = [outputs[i].to_host() for i in range(1, len(outputs))]
        return logits, new_kv_cache

    def sample_next_token(self, logits: np.ndarray, temperature: float) -> int:
        """Sample next token from logits"""
        last_logits = logits[0, -1, :]
        if temperature <= 0:
            return int(np.argmax(last_logits))
        scaled = last_logits / temperature
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled) / np.sum(np.exp(scaled))
        return int(np.random.choice(len(probs), p=probs))

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        """Generate response with KV cache"""
        inputs = self.prepare_inputs(image, question)
        seq_len = inputs["input_ids"].shape[1]

        # Prefill
        t0 = time.perf_counter()
        logits, kv_cache = self.forward_prefill(inputs)
        prefill_time = time.perf_counter() - t0
        print(f"  Prefill: {prefill_time:.2f}s, seq_len={seq_len}")

        # First token
        generated_ids = []
        next_token = self.sample_next_token(logits, temperature)
        generated_ids.append(next_token)

        # Decode loop with KV cache
        if self.decode_fn is not None and max_new_tokens > 1:
            t0 = time.perf_counter()
            for step in range(1, max_new_tokens):
                if next_token == self.eos_token_id:
                    break

                # Prepare decode inputs
                input_ids = np.array([[next_token]], dtype=np.int64)
                position_ids = np.array([[seq_len + step - 1]], dtype=np.int64)
                attention_mask = np.ones((1, seq_len + step), dtype=np.int64)

                logits, kv_cache = self.forward_decode(
                    input_ids, attention_mask, position_ids, kv_cache)
                next_token = self.sample_next_token(logits, temperature)
                generated_ids.append(next_token)

            decode_time = time.perf_counter() - t0
            tokens_decoded = len(generated_ids) - 1
            if tokens_decoded > 0:
                print(f"  Decode: {decode_time:.2f}s, {tokens_decoded} tokens, "
                      f"{tokens_decoded/decode_time:.1f} tok/s")
        elif max_new_tokens > 1 and self.decode_fn is None:
            print("  Warning: decode.vmfb not found, only 1 token generated")

        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True)
        return response

    def describe_image(self, image_path: str, question: str = "Describe this image."):
        """Describe an image"""
        image = Image.open(image_path).convert("RGB")
        print(f"\n[Image] {image_path}")
        print(f"[Question] {question}")
        response = self.generate(
            image, question, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
        print(f"[Response] {response}")
        return response


def create_demo_image(path: str) -> str:
    """Create demo image"""
    img = Image.new("RGB", (200, 200), color="skyblue")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill="yellow", outline="orange")
    draw.rectangle([80, 160, 120, 200], fill="brown")
    draw.text((60, 10), "Demo", fill="black")
    img.save(path)
    return path


def main():
    print("=" * 60)
    print("Qwen3-VL-2B IREE Inference")
    print("=" * 60)

    # Initialize runner
    runner = Qwen3VLIREERunner()

    # Demo image
    demo_img = str(IREE_DIR / "demo_image.png")
    if not os.path.exists(demo_img):
        create_demo_image(demo_img)
        print(f"\nCreated demo image: {demo_img}")

    # Run inference
    image_path = IMAGE_PATH if IMAGE_PATH else demo_img
    runner.describe_image(image_path, QUESTION)

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive mode:")
    print("  Format: image_path | question")
    print("  Example: cat.jpg | What is this?")
    print("  Just question to use current image")
    print("  Ctrl+C to exit")
    print("=" * 60)

    current_image = image_path

    while True:
        try:
            inp = input("\n> ").strip()
            if not inp:
                continue

            if "|" in inp:
                parts = inp.split("|", 1)
                img_path = parts[0].strip()
                question = parts[1].strip()
                assert os.path.exists(img_path), f"Image not found: {img_path}"
                current_image = img_path
            else:
                question = inp

            runner.describe_image(current_image, question)

        except KeyboardInterrupt:
            print("\nExit")
            break


if __name__ == "__main__":
    main()
