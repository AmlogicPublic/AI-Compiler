import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model


def load_processor(model_name: str):
    from transformers import AutoProcessor

    local_dir = download_model(model_name)
    return AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)


def prepare_inputs(processor, image: Image.Image, question: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    return {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
        "pixel_values": inputs["pixel_values"].numpy(),
        "image_grid_thw": inputs["image_grid_thw"].numpy(),
    }


def sample_next_token(logits: np.ndarray, temperature: float) -> int:
    last_logits = logits[0, -1, :]
    if temperature <= 0:
        return int(np.argmax(last_logits))
    scaled = last_logits / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return int(np.random.choice(len(probs), p=probs))


def create_demo_image(path: str) -> str:
    img = Image.new("RGB", (200, 200), color="skyblue")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill="yellow", outline="orange")
    draw.rectangle([80, 160, 120, 200], fill="brown")
    draw.text((60, 10), "Demo", fill="black")
    img.save(path)
    return path


def generate_response(runner, image, question: str, max_new_tokens: int, temperature: float) -> str:
    inputs = prepare_inputs(runner.processor, image, question)
    seq_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    logits, kv_cache = runner.forward_prefill(inputs)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.2f}s, seq_len={seq_len}")

    generated_ids = []
    next_token = sample_next_token(logits, temperature)
    generated_ids.append(next_token)

    if max_new_tokens > 1:
        t0 = time.perf_counter()
        for step in range(1, max_new_tokens):
            if next_token == runner.eos_token_id:
                break

            input_ids = np.array([[next_token]], dtype=np.int64)
            position_ids = np.array([[seq_len + step - 1]], dtype=np.int64)
            attention_mask = np.ones((1, seq_len + step), dtype=np.int64)

            logits, kv_cache = runner.forward_decode(input_ids, attention_mask, position_ids, kv_cache)
            next_token = sample_next_token(logits, temperature)
            generated_ids.append(next_token)

        decode_time = time.perf_counter() - t0
        tokens_decoded = len(generated_ids) - 1
        if tokens_decoded > 0:
            print(f"  Decode: {decode_time:.2f}s, {tokens_decoded} tokens, {tokens_decoded / decode_time:.1f} tok/s")

    return runner.tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_interactive_loop(
    runner,
    *,
    title: str,
    demo_image_dir: Path,
    image_path: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
):
    print("=" * 60)
    print(title)
    print("=" * 60)

    demo_img = str(demo_image_dir / "demo_image.png")
    if not os.path.exists(demo_img):
        create_demo_image(demo_img)

    current_image = image_path if image_path else demo_img
    image = Image.open(current_image).convert("RGB")

    print(f"\n[Image] {current_image}")
    print(f"[Question] {question}")
    response = generate_response(runner, image, question, max_new_tokens=max_new_tokens, temperature=temperature)
    print(f"[Response] {response}")

    print("\n" + "=" * 60)
    print("Interactive mode:")
    print("  Format: image_path | question")
    print("  Example: cat.jpg | What is this?")
    print("  Just question to use current image")
    print("  Ctrl+C to exit")
    print("=" * 60)

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

            image = Image.open(current_image).convert("RGB")
            print(f"\n[Image] {current_image}")
            print(f"[Question] {question}")
            response = generate_response(runner, image, question, max_new_tokens=max_new_tokens, temperature=temperature)
            print(f"[Response] {response}")
        except KeyboardInterrupt:
            print("\nExit")
            break
