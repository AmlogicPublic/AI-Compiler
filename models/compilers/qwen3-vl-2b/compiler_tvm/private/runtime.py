import tvm

from stage1_run.settings import IMAGE_PATH, MAX_NEW_TOKENS, MODEL_NAME, QUESTION, TEMPERATURE
from stage1_run.vl_runtime import load_processor, run_interactive_loop

from private.backend import (
    COMPILED_DIR,
    load_params_for_module,
    load_tvm_module,
    to_tvm_decode_inputs,
    to_tvm_prefill_inputs,
    unpack_tvm_outputs,
)


class Qwen3VLTVMRunner:
    def __init__(self):
        print("Loading TVM modules...")
        self.device = tvm.cpu()
        self.processor = load_processor(MODEL_NAME)
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

        prefill_lib = COMPILED_DIR / "prefill.so"
        decode_lib = COMPILED_DIR / "decode.so"
        self.prefill_vm = load_tvm_module(prefill_lib, self.device)
        self.decode_vm = load_tvm_module(decode_lib, self.device)
        self.prefill_params = load_params_for_module(prefill_lib, self.device)
        self.decode_params = load_params_for_module(decode_lib, self.device)

        print("TVM runner ready")

    def forward_prefill(self, inputs: dict):
        input_ids_tvm, attention_mask_tvm, pixel_values_tvm = to_tvm_prefill_inputs(inputs, self.device)
        outputs = self.prefill_vm["main"](
            input_ids_tvm,
            attention_mask_tvm,
            pixel_values_tvm,
            *self.prefill_params,
        )
        return unpack_tvm_outputs(outputs)

    def forward_decode(self, input_ids, attention_mask, position_ids, kv_cache):
        input_ids_tvm, attention_mask_tvm, position_ids_tvm, kv_cache_tvm = to_tvm_decode_inputs(
            input_ids,
            attention_mask,
            position_ids,
            kv_cache,
            self.device,
        )
        outputs = self.decode_vm["main"](
            input_ids_tvm,
            attention_mask_tvm,
            position_ids_tvm,
            *kv_cache_tvm,
            *self.decode_params,
        )
        return unpack_tvm_outputs(outputs)


def main():
    runner = Qwen3VLTVMRunner()
    run_interactive_loop(
        runner,
        title="Qwen3-VL-2B TVM Inference",
        demo_image_dir=COMPILED_DIR.parent,
        image_path=IMAGE_PATH,
        question=QUESTION,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
