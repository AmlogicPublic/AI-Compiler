"""SmolLM2-135M TVM Runtime 推理
使用编译好的 TVM library 运行推理
"""
import sys
import importlib.util
from pathlib import Path

import numpy as np
import tvm
from tvm import relax

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from huggingface_run.shared import download_model

TVM_DIR = Path(__file__).parent
STAGE0_EXPORT_DIR = TVM_DIR / "stage0_export"
COMPILED_DIR = TVM_DIR / "stage0_export" / "compiled"


def _load_stage_common_module():
    stage_common_path = STAGE0_EXPORT_DIR / "stage_common.py"
    assert stage_common_path.exists(), f"Missing stage_common.py: {stage_common_path}"
    spec = importlib.util.spec_from_file_location("smollm2_tvm_stage_common", stage_common_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGE_COMMON = _load_stage_common_module()


def load_params_for_module(lib_path: Path, device) -> list:
    """Load external params by module IR signature order."""
    params_path = COMPILED_DIR / "params.npz"
    ir_path = lib_path.with_suffix(".txt")
    assert params_path.exists(), f"Params not found: {params_path}\nRun 'python stage0_export/export.py' first"
    assert ir_path.exists(), f"IR not found: {ir_path}\nRun 'python stage0_export/export.py' first"
    return STAGE_COMMON.load_params_for_tvm(params_path, ir_path, device)

# ===================== User Macros =====================
MODEL_NAME = "smollm2-135m"
PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
# =======================================================
PREFILL_SEQ_LEN = 32
DECODE_PAST_LEN = 100
DECODE_TOTAL_LEN = DECODE_PAST_LEN + 1


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

        # Load params from params.npz in IR signature order
        print("Loading params from params.npz...")
        self.prefill_params = load_params_for_module(prefill_lib, self.device)
        self.decode_params = load_params_for_module(decode_lib, self.device)
        print(f"  Prefill params: {len(self.prefill_params)}, Decode params: {len(self.decode_params)}")

        print("TVM runner ready")

    def prepare_inputs(self, prompt: str):
        """Prepare inputs for TVM inference"""
        # Must pad to 32 to match compiled prefill model shape
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=PREFILL_SEQ_LEN,
        )
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        assert prompt_len > 0, "Prompt is empty after tokenization"
        return {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
            "prompt_len": prompt_len,
        }

    def unpack_tvm_outputs(self, outputs):
        """Unpack TVM VM outputs for both NDArray and tuple-like ADT."""
        if hasattr(outputs, "numpy"):
            return outputs.numpy(), []

        assert len(outputs) > 0, "VM main output is empty"

        first = outputs[0]
        logits = first.numpy() if hasattr(first, "numpy") else first.asnumpy()

        kv_cache = []
        for i in range(1, len(outputs)):
            out = outputs[i]
            kv_cache.append(out.numpy() if hasattr(out, "numpy") else out.asnumpy())
        return logits, kv_cache

    def init_decode_kv_cache(self, prefill_kv_cache: list[np.ndarray], prompt_len: int) -> list[np.ndarray]:
        """Convert prefill KV (len=32) to decode KV (len=100) with right alignment."""
        assert prompt_len <= PREFILL_SEQ_LEN
        decode_kv_cache = []
        for kv in prefill_kv_cache:
            assert kv.ndim == 4
            assert kv.shape[2] == PREFILL_SEQ_LEN
            kv_valid = kv[:, :, :prompt_len, :]
            pad_len = DECODE_PAST_LEN - prompt_len
            assert pad_len >= 0
            if pad_len == 0:
                decode_kv_cache.append(kv_valid)
            else:
                pad = np.zeros((kv.shape[0], kv.shape[1], pad_len, kv.shape[3]), dtype=kv.dtype)
                decode_kv_cache.append(np.concatenate([pad, kv_valid], axis=2))
        return decode_kv_cache

    def build_decode_attention_mask(self, past_valid_len: int) -> np.ndarray:
        """Build static decode attention mask with shape (1, 101)."""
        assert 0 < past_valid_len <= DECODE_PAST_LEN
        mask = np.zeros((1, DECODE_TOTAL_LEN), dtype=np.int64)
        mask[0, DECODE_PAST_LEN - past_valid_len : DECODE_PAST_LEN] = 1
        mask[0, DECODE_PAST_LEN] = 1
        return mask

    def clip_decode_kv_cache(self, decode_outputs_kv: list[np.ndarray]) -> list[np.ndarray]:
        """Decode returns len=101 cache; next decode input must be len=100."""
        kv_cache = []
        for kv in decode_outputs_kv:
            assert kv.shape[2] == DECODE_TOTAL_LEN
            kv_cache.append(kv[:, :, -DECODE_PAST_LEN:, :])
        return kv_cache

    def forward_prefill(self, inputs: dict):
        """Prefill: prompt → logits + KV cache"""
        input_ids_tvm = tvm.runtime.tensor(inputs["input_ids"], device=self.device)
        attention_mask_tvm = tvm.runtime.tensor(inputs["attention_mask"], device=self.device)

        # Weights are passed as additional arguments after inputs
        outputs = self.prefill_vm["main"](input_ids_tvm, attention_mask_tvm, *self.prefill_params)
        return self.unpack_tvm_outputs(outputs)

    def forward_decode(self, input_ids, attention_mask, position_ids, cache_position, kv_cache):
        """Decode: 1 token + KV cache → logits + new KV cache"""
        input_ids_tvm = tvm.runtime.tensor(input_ids, device=self.device)
        attention_mask_tvm = tvm.runtime.tensor(attention_mask, device=self.device)
        position_ids_tvm = tvm.runtime.tensor(position_ids, device=self.device)
        cache_position_tvm = tvm.runtime.tensor(cache_position, device=self.device)
        kv_cache_tvm = [tvm.runtime.tensor(kv, device=self.device) for kv in kv_cache]

        # Weights are passed as additional arguments after inputs
        outputs = self.decode_vm["main"](
            input_ids_tvm, attention_mask_tvm, position_ids_tvm, cache_position_tvm, *kv_cache_tvm, *self.decode_params
        )
        return self.unpack_tvm_outputs(outputs)

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
        prompt_len = inputs["prompt_len"]
        assert prompt_len <= seq_len

        # Prefill
        t0 = time.perf_counter()
        logits, kv_cache = self.forward_prefill(inputs)
        prefill_time = time.perf_counter() - t0
        print(f"  Prefill: {prefill_time:.2f}s, padded_seq_len={seq_len}, prompt_len={prompt_len}")

        # First token
        generated_ids = []
        next_token = self.sample_next_token(logits[:, :prompt_len, :], temperature)
        generated_ids.append(next_token)

        if stream:
            print("[Generated] ", end="", flush=True)
            print(self.tokenizer.decode([next_token], skip_special_tokens=True), end="", flush=True)

        # Decode loop with KV cache
        if max_new_tokens > 1 and len(kv_cache) > 0:
            t0 = time.perf_counter()
            kv_cache = self.init_decode_kv_cache(kv_cache, prompt_len)
            past_valid_len = prompt_len
            for step in range(1, max_new_tokens):
                if next_token == self.eos_token_id:
                    break

                # Prepare decode inputs
                cur_pos = prompt_len + step - 1
                input_ids = np.array([[next_token]], dtype=np.int64)
                position_ids = np.array([[cur_pos]], dtype=np.int64)
                cache_position = np.array([cur_pos], dtype=np.int64)
                attention_mask = self.build_decode_attention_mask(past_valid_len)

                logits, decode_outputs_kv = self.forward_decode(
                    input_ids, attention_mask, position_ids, cache_position, kv_cache
                )
                kv_cache = self.clip_decode_kv_cache(decode_outputs_kv)
                past_valid_len = min(past_valid_len + 1, DECODE_PAST_LEN)
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
