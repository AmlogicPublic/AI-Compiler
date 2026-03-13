from stage1_run.settings import IMAGE_PATH, MAX_NEW_TOKENS, MODEL_NAME, QUESTION, TEMPERATURE
from stage1_run.vl_runtime import load_processor, run_interactive_loop

from private.backend import COMPILED_DIR, load_iree_module


class Qwen3VLIREERunner:
    def __init__(self):
        print("Loading IREE modules...")
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


def main():
    runner = Qwen3VLIREERunner()
    run_interactive_loop(
        runner,
        title="Qwen3-VL-2B IREE Inference",
        demo_image_dir=COMPILED_DIR.parent,
        image_path=IMAGE_PATH,
        question=QUESTION,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
