import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    local_dir = download_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_inputs(tokenizer, prompt: str, prefill_seq_len: int):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=prefill_seq_len,
    )
    return {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }


def sample_next_token(logits: np.ndarray, temperature: float) -> int:
    last_logits = logits[0, -1, :]
    if temperature <= 0:
        return int(np.argmax(last_logits))
    scaled = last_logits / temperature
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled) / np.sum(np.exp(scaled))
    return int(np.random.choice(len(probs), p=probs))


def generate_text(runner, prompt: str, max_new_tokens: int, temperature: float, stream: bool) -> str:
    t_generate = time.perf_counter()
    inputs = prepare_inputs(runner.tokenizer, prompt, runner.prefill_seq_len)
    seq_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    logits, kv_cache = runner.forward_prefill(inputs)
    prefill_time = time.perf_counter() - t0
    print(f"  Prefill: {prefill_time:.2f}s, seq_len={seq_len}")

    generated_ids = []
    next_token = sample_next_token(logits, temperature)
    generated_ids.append(next_token)

    if stream:
        print("[Generated] ", end="", flush=True)
        print(runner.tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

    decode_time = None
    tokens_decoded = 0
    if max_new_tokens > 1:
        assert len(kv_cache) > 0
        t0 = time.perf_counter()
        for step in range(1, max_new_tokens):
            if next_token == runner.eos_token_id:
                break

            cur_pos = seq_len + step - 1
            input_ids = np.array([[next_token]], dtype=np.int64)
            position_ids = np.array([[cur_pos]], dtype=np.int64)
            cache_position = np.array([cur_pos], dtype=np.int64)

            logits, kv_cache = runner.forward_decode(input_ids, position_ids, cache_position, kv_cache)
            next_token = sample_next_token(logits, temperature)
            generated_ids.append(next_token)

            if stream:
                print(runner.tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

        decode_time = time.perf_counter() - t0
        tokens_decoded = len(generated_ids) - 1

    if stream:
        print()
    if decode_time is not None:
        speed = tokens_decoded / decode_time if decode_time > 0 else 0.0
        print(f"  Decode: {decode_time:.2f}s, {tokens_decoded} tokens, {speed:.1f} tok/s")

    total_time = time.perf_counter() - t_generate
    total_tokens = len(generated_ids)
    total_speed = total_tokens / total_time if total_time > 0 else 0.0
    print(f"  Total: {total_time:.2f}s, {total_tokens} tokens, {total_speed:.1f} tok/s, TTFT={prefill_time:.2f}s")

    return runner.tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_interactive_loop(runner, title: str, init_prompt: str, max_new_tokens: int, temperature: float):
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"\n[Prompt] {init_prompt}")
    generate_text(runner, init_prompt, max_new_tokens=max_new_tokens, temperature=temperature, stream=True)

    print("\n" + "=" * 60)
    print("Interactive mode:")
    print("  Enter prompt, Ctrl+C to exit")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                continue
            print(f"\n[Prompt] {prompt}")
            generate_text(runner, prompt, max_new_tokens=max_new_tokens, temperature=temperature, stream=True)
        except KeyboardInterrupt:
            print("\nExit")
            break
