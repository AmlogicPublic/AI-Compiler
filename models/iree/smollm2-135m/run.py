"""SmolLM2-135M IREE Runtime 推理
使用编译好的 vmfb 文件运行推理
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_ROOT = REPO_ROOT / "models"
sys.path.insert(0, str(MODELS_ROOT))

from run.shared import download_model
import numpy as np
import time

IREE_DIR = Path(__file__).parent
COMPILED_DIR = IREE_DIR / "stage0_export" / "compiled"

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


def load_iree_module(vmfb_path: Path, params_path: Path | None = None):
    """Load compiled IREE module"""
    import iree.runtime as ireert

    assert vmfb_path.exists(), f"VMFB not found: {vmfb_path}\nRun 'python export.py' first"

    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)

    if params_path is not None and params_path.exists():
        param_index = ireert.ParameterIndex()
        param_index.load(str(params_path))
        param_provider = param_index.create_provider(scope="model")
        param_module = ireert.create_io_parameters_module(ctx.instance, param_provider)
        ctx.add_vm_module(param_module)

    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)

    return ctx


class SmolLM2IREERunner:
    """IREE-based SmolLM2 runner with KV Cache support"""

    def __init__(self):
        print("Loading IREE modules...")

        self.tokenizer = load_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id

        params_path = COMPILED_DIR / "smollm2_135m.irpa"

        # Load prefill module (prompt → logits + KV cache)
        prefill_vmfb = COMPILED_DIR / "prefill.vmfb"
        assert prefill_vmfb.exists(), f"Run 'python stage0_export/export.py' first: {prefill_vmfb}"
        self.prefill_ctx = load_iree_module(prefill_vmfb, params_path)
        self.prefill_fn = self.prefill_ctx.modules.module["main"]

        # Load decode module (1 token + KV cache → logits + new KV cache)
        decode_vmfb = COMPILED_DIR / "decode.vmfb"
        assert decode_vmfb.exists(), f"Run 'python stage0_export/export.py' first: {decode_vmfb}"
        self.decode_ctx = load_iree_module(decode_vmfb, params_path)
        self.decode_fn = self.decode_ctx.modules.module["main"]

        print("IREE runner ready")

    def prepare_inputs(self, prompt: str):
        """Prepare inputs for IREE inference"""
        # Must pad to 32 to match compiled prefill model shape
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=32)
        return {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }

    def forward_prefill(self, inputs: dict):
        """Prefill: prompt → logits + KV cache"""
        outputs = self.prefill_fn(
            inputs["input_ids"],
            inputs["attention_mask"],
        )
        # outputs: (logits, k0, v0, k1, v1, ...)
        logits = outputs[0].to_host()
        kv_cache = [outputs[i].to_host() for i in range(1, len(outputs))]
        return logits, kv_cache

    def forward_decode(self, input_ids, attention_mask, position_ids, cache_position, kv_cache):
        """Decode: 1 token + KV cache → logits + new KV cache"""
        outputs = self.decode_fn(input_ids, attention_mask, position_ids, cache_position, *kv_cache)
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
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Generate response with KV cache"""
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
        if max_new_tokens > 1:
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
    print("SmolLM2-135M IREE Inference")
    print("=" * 60)

    # Initialize runner
    runner = SmolLM2IREERunner()

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
