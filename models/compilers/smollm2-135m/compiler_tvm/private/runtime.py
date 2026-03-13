import sys
from pathlib import Path

import tvm

MODEL_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODEL_ROOT))

from stage1_run.settings import MAX_NEW_TOKENS, MODEL_NAME, PREFILL_SEQ_LEN, PROMPT, TEMPERATURE
from stage1_run.text_runtime import load_tokenizer, run_interactive_loop

from private.backend import (
    COMPILED_DIR,
    load_params_for_module,
    load_tvm_module,
    to_tvm_decode_inputs,
    unpack_tvm_outputs,
)


class SmolLM2TVMRunner:
    def __init__(self):
        print("Loading TVM modules...")
        self.prefill_seq_len = PREFILL_SEQ_LEN
        self.device = tvm.cpu()
        self.tokenizer = load_tokenizer(MODEL_NAME)
        self.eos_token_id = self.tokenizer.eos_token_id

        prefill_lib = COMPILED_DIR / "prefill.so"
        decode_lib = COMPILED_DIR / "decode.so"
        self.prefill_vm = load_tvm_module(prefill_lib, self.device)
        self.decode_vm = load_tvm_module(decode_lib, self.device)
        self.prefill_params = load_params_for_module(prefill_lib, self.device)
        self.decode_params = load_params_for_module(decode_lib, self.device)

        print("TVM runner ready")

    def forward_prefill(self, inputs: dict):
        input_ids_tvm = tvm.runtime.tensor(inputs["input_ids"], device=self.device)
        attention_mask_tvm = tvm.runtime.tensor(inputs["attention_mask"], device=self.device)
        outputs = self.prefill_vm["main"](input_ids_tvm, attention_mask_tvm, *self.prefill_params)
        return unpack_tvm_outputs(outputs)

    def forward_decode(self, input_ids, position_ids, cache_position, kv_cache):
        input_ids_tvm, position_ids_tvm, cache_position_tvm, kv_cache_tvm = to_tvm_decode_inputs(
            input_ids,
            position_ids,
            cache_position,
            kv_cache,
            self.device,
        )
        outputs = self.decode_vm["main"](
            input_ids_tvm,
            position_ids_tvm,
            cache_position_tvm,
            *kv_cache_tvm,
            *self.decode_params,
        )
        return unpack_tvm_outputs(outputs)


def main():
    runner = SmolLM2TVMRunner()
    run_interactive_loop(
        runner,
        title="SmolLM2-135M TVM Inference",
        init_prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
