import sys
import time
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from stage1_run.settings import IMAGE_PATH, MAX_NEW_TOKENS, MODEL_NAME, QUESTION, TEMPERATURE
from stage1_run.vl_runtime import load_processor, prepare_inputs, run_interactive_loop, sample_next_token

from private.backend import COMPILED_DIR, MODEL_ROOT, load_iree_module


class Qwen3VLIREERunner:
    def __init__(self):
        print("Loading IREE modules...")
        self.model_root = MODEL_ROOT
        self.processor = load_processor(MODEL_NAME)
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

        self.prefill_fn = load_iree_module(COMPILED_DIR / "prefill.vmfb").modules.module["main"]
        self.decode_fn = load_iree_module(COMPILED_DIR / "decode.vmfb").modules.module["main"]

        print("IREE runner ready")

    def forward_prefill(self, inputs: dict):
        outputs = self.prefill_fn(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        logits = outputs[0].to_host()
        kv_cache = [outputs[i].to_host() for i in range(1, len(outputs))]
        return logits, kv_cache

    def forward_decode(self, input_ids, attention_mask, position_ids, kv_cache):
        outputs = self.decode_fn(input_ids, attention_mask, position_ids, *kv_cache)
        logits = outputs[0].to_host()
        new_kv_cache = [outputs[i].to_host() for i in range(1, len(outputs))]
        return logits, new_kv_cache

    def generate(self, image, question: str, max_new_tokens: int, temperature: float) -> str:
        inputs = prepare_inputs(self.processor, image, question)
        seq_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        logits, kv_cache = self.forward_prefill(inputs)
        prefill_time = time.perf_counter() - t0
        print(f"  Prefill: {prefill_time:.2f}s, seq_len={seq_len}")

        generated_ids = []
        next_token = sample_next_token(logits, temperature)
        generated_ids.append(next_token)

        if max_new_tokens > 1:
            t0 = time.perf_counter()
            for step in range(1, max_new_tokens):
                if next_token == self.eos_token_id:
                    break

                input_ids = np.array([[next_token]], dtype=np.int64)
                position_ids = np.array([[seq_len + step - 1]], dtype=np.int64)
                attention_mask = np.ones((1, seq_len + step), dtype=np.int64)

                logits, kv_cache = self.forward_decode(input_ids, attention_mask, position_ids, kv_cache)
                next_token = sample_next_token(logits, temperature)
                generated_ids.append(next_token)

            decode_time = time.perf_counter() - t0
            tokens_decoded = len(generated_ids) - 1
            if tokens_decoded > 0:
                print(f"  Decode: {decode_time:.2f}s, {tokens_decoded} tokens, {tokens_decoded / decode_time:.1f} tok/s")

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    runner = Qwen3VLIREERunner()
    run_interactive_loop(
        runner,
        image_path=IMAGE_PATH,
        question=QUESTION,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
