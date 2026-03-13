from shared.stage1_run.settings import MAX_NEW_TOKENS, MODEL_NAME, PREFILL_SEQ_LEN, PROMPT, TEMPERATURE
from shared.stage1_run.text_runtime import load_tokenizer, run_interactive_loop

from private.backend import COMPILED_DIR, load_iree_module


class SmolLM2IREERunner:
    def __init__(self):
        print("Loading IREE modules...")
        self.prefill_seq_len = PREFILL_SEQ_LEN
        self.tokenizer = load_tokenizer(MODEL_NAME)
        self.eos_token_id = self.tokenizer.eos_token_id

        params_path = COMPILED_DIR / "smollm2_135m.irpa"
        self.prefill_fn = load_iree_module(COMPILED_DIR / "prefill.vmfb", params_path).modules.module["main"]
        self.decode_fn = load_iree_module(COMPILED_DIR / "decode.vmfb", params_path).modules.module["main"]

        print("IREE runner ready")

    def forward_prefill(self, inputs: dict):
        outputs = self.prefill_fn(inputs["input_ids"], inputs["attention_mask"])
        logits = outputs[0].to_host()
        kv_cache = [outputs[i] for i in range(1, len(outputs))]
        return logits, kv_cache

    def forward_decode(self, input_ids, position_ids, cache_position, kv_cache):
        outputs = self.decode_fn(input_ids, position_ids, cache_position, *kv_cache)
        logits = outputs[0].to_host()
        new_kv_cache = [outputs[i] for i in range(1, len(outputs))]
        return logits, new_kv_cache


def main():
    runner = SmolLM2IREERunner()
    run_interactive_loop(
        runner,
        title="SmolLM2-135M IREE Inference",
        init_prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
