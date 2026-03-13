"""SmolLM2-135M TVM Runtime 推理
使用编译好的 TVM library 运行推理
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import tvm
from tvm import relax

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model

TVM_DIR = Path(__file__).parent
COMPILED_DIR = TVM_DIR / "stage0_export" / "compiled"


def load_params_from_onnx(onnx_path: Path, device) -> list:
    """Load params from ONNX in the order they appear as initializers."""
    model = onnx.load(str(onnx_path))
    params = []
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        params.append(tvm.nd.array(arr, device))
    return params

# ===================== User Macros =====================
MODEL_NAME = "smollm2-135m"
PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
# =======================================================


def load_tokenizer():
    """Load SmolLM2 tokenizer"""
    from transformers import AutoTokenizer

    local_dir = download_model(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_tvm_module(lib_path: Path, device):
    """Load compiled TVM module"""
    assert lib_path.exists(), f"Library not found: {lib_path}\nRun 'python export.py' first"
    ex = tvm.runtime.load_module(str(lib_path))
    vm = relax.VirtualMachine(ex, device)
    return vm


class SmolLM2TVMRunner:
    """TVM-based SmolLM2 runner with KV Cache support"""

    def __init__(self):
        print("Loading TVM modules...")

        self.device = tvm.cpu()
        self.tokenizer = load_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id

        # Load prefill module (prompt → logits + KV cache)
        prefill_lib = COMPILED_DIR / "prefill.so"
        assert prefill_lib.exists(), f"Run 'python stage0_export/export.py' first: {prefill_lib}"
        self.prefill_vm = load_tvm_module(prefill_lib, self.device)

        # Load decode module (1 token + KV cache → logits + new KV cache)
        decode_lib = COMPILED_DIR / "decode.so"
        assert decode_lib.exists(), f"Run 'python stage0_export/export.py' first: {decode_lib}"
        self.decode_vm = load_tvm_module(decode_lib, self.device)

        # Load params from ONNX (weights are external now)
        prefill_onnx = COMPILED_DIR / "prefill.onnx"
        decode_onnx = COMPILED_DIR / "decode.onnx"
        print("Loading params from ONNX...")
        self.prefill_params = load_params_from_onnx(prefill_onnx, self.device)
        self.decode_params = load_params_from_onnx(decode_onnx, self.device)
        print(f"  Prefill params: {len(self.prefill_params)}, Decode params: {len(self.decode_params)}")

        print("TVM runner ready")

    def prepare_inputs(self, prompt: str):
        """Prepare inputs for TVM inference"""
        # Must pad to 32 to match compiled prefill model shape
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=32)
        return {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }

    def forward_prefill(self, inputs: dict):
        """Prefill: prompt → logits + KV cache"""
        input_ids_tvm = tvm.runtime.from_dlpack(inputs["input_ids"])
        attention_mask_tvm = tvm.runtime.from_dlpack(inputs["attention_mask"])

        # Weights are passed as additional arguments after inputs
        outputs = self.prefill_vm["main"](input_ids_tvm, attention_mask_tvm, *self.prefill_params)

        # outputs: (logits, k0, v0, k1, v1, ...)
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0].numpy()
            kv_cache = [outputs[i].numpy() for i in range(1, len(outputs))]
        else:
            logits = outputs.numpy()
            kv_cache = []
        return logits, kv_cache

    def forward_decode(self, input_ids, attention_mask, position_ids, cache_position, kv_cache):
        """Decode: 1 token + KV cache → logits + new KV cache"""
        input_ids_tvm = tvm.runtime.from_dlpack(input_ids)
        attention_mask_tvm = tvm.runtime.from_dlpack(attention_mask)
        position_ids_tvm = tvm.runtime.from_dlpack(position_ids)
        cache_position_tvm = tvm.runtime.from_dlpack(cache_position)
        kv_cache_tvm = [tvm.runtime.from_dlpack(kv) for kv in kv_cache]

        # Weights are passed as additional arguments after inputs
        outputs = self.decode_vm["main"](
            input_ids_tvm, attention_mask_tvm, position_ids_tvm, cache_position_tvm, *kv_cache_tvm, *self.decode_params
        )

        if isinstance(outputs, (list, tuple)):
            logits = outputs[0].numpy()
            new_kv_cache = [outputs[i].numpy() for i in range(1, len(outputs))]
        else:
            logits = outputs.numpy()
            new_kv_cache = []
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
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Generate response with KV cache"""
        import time

        inputs = self.prepare_inputs(prompt)
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

        if stream:
            print("[Generated] ", end="", flush=True)
            print(self.tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

        # Decode loop with KV cache
        if max_new_tokens > 1 and len(kv_cache) > 0:
            t0 = time.perf_counter()
            for step in range(1, max_new_tokens):
                if next_token == self.eos_token_id:
                    break

                # Prepare decode inputs
                cur_pos = seq_len + step - 1
                input_ids = np.array([[next_token]], dtype=np.int64)
                position_ids = np.array([[cur_pos]], dtype=np.int64)
                cache_position = np.array([cur_pos], dtype=np.int64)
                attention_mask = np.ones((1, seq_len + step), dtype=np.int64)

                logits, kv_cache = self.forward_decode(
                    input_ids, attention_mask, position_ids, cache_position, kv_cache
                )
                next_token = self.sample_next_token(logits, temperature)
                generated_ids.append(next_token)

                if stream:
                    print(self.tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

            decode_time = time.perf_counter() - t0
            tokens_decoded = len(generated_ids) - 1
            if stream:
                print()  # newline after streaming
            if tokens_decoded > 0:
                print(
                    f"  Decode: {decode_time:.2f}s, {tokens_decoded} tokens, "
                    f"{tokens_decoded/decode_time:.1f} tok/s"
                )

        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response


def main():
    print("=" * 60)
    print("SmolLM2-135M TVM Inference")
    print("=" * 60)

    # Initialize runner
    runner = SmolLM2TVMRunner()

    # Run inference
    print(f"\n[Prompt] {PROMPT}")
    response = runner.generate(PROMPT, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE)
    print(f"[Generated] {response}")

    # Interactive mode
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
            runner.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )
        except KeyboardInterrupt:
            print("\nExit")
            break


if __name__ == "__main__":
    main()
