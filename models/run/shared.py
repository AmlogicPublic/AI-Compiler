"""公用配置与模型加载"""
import os
from pathlib import Path
from threading import Thread
import torch

MODELS_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(MODELS_ROOT / "huggingface")

MODELS = {
    "bge-micro-v2": "TaylorAI/bge-micro-v2",                    # ~17M embedding
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",               # ~135M text
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",                 # ~2.5B multimodal
    "gemma-2-2b": "google/gemma-2-2b-it",                       # ~2.6B text
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",           # ~3.8B text
    "sd-turbo": "stabilityai/sd-turbo",                         # ~1B diffusion (图像生成)
}

# Diffusion 模型列表 (用 diffusers 库，不是 transformers)
DIFFUSION_MODELS = {"sd-turbo"}


def _resolve_local_model_dir(name, check_file):
    candidates = [
        MODELS_ROOT / "huggingface" / name,
        MODELS_ROOT / name,
    ]
    for path in candidates:
        if (path / check_file).exists():
            return str(path)
    return str(candidates[0])


def download_model(name):
    """优先使用本地模型目录，缺失时下载到 models/huggingface/name"""
    from huggingface_hub import snapshot_download

    # diffusion 用 model_index.json, transformers 用 config.json
    check_file = "model_index.json" if name in DIFFUSION_MODELS else "config.json"
    local_dir = _resolve_local_model_dir(name, check_file)

    if os.path.exists(os.path.join(local_dir, check_file)):
        print(f"使用本地模型: {local_dir}")
        return local_dir

    repo_id = MODELS[name]
    print(f"下载中: {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("下载完成")
    assert os.path.exists(os.path.join(local_dir, check_file)), f"缺少模型文件: {os.path.join(local_dir, check_file)}"
    return local_dir


def generate_with_stream(
    model,
    tokenizer,
    inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=None,
):
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    if pad_token_id is None and hasattr(tokenizer, "eos_token_id"):
        pad_token_id = tokenizer.eos_token_id

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        streamer=streamer,
    )
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        print(token, end="", flush=True)
        response += token
    print()

    thread.join()
    return response


class StreamingChatSession:
    def __init__(self, model, tokenizer, default_max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.default_max_new_tokens = default_max_new_tokens
        self.history = []

    def chat(self, user_input, max_new_tokens=None):
        self.history.append({"role": "user", "content": user_input})
        prompt = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        response = generate_with_stream(
            self.model,
            self.tokenizer,
            inputs,
            max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.history = []


def load_model(name, load_tokenizer=True, load_processor=False):
    """加载模型, 返回 (model, config, tokenizer/processor)"""
    import time
    
    local_dir = download_model(name)
    t0 = time.perf_counter()
    
    # Diffusion 模型: 返回 UNet
    if name in DIFFUSION_MODELS:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(local_dir, torch_dtype=torch.float32)
        model = pipe.unet
        config = model.config
        elapsed = time.perf_counter() - t0
        print(f"加载模型: {elapsed:.2f}s, 参数量: {sum(p.numel() for p in model.parameters()):,}")
        model.eval()
        return model, config, None
    
    else:
        # Transformers 模型
        from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer
        config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        arch = config.architectures[0] if hasattr(config, 'architectures') else ""
        
        needs_remote_code = "Qwen" in arch and "VL" in arch
        
        # tokenizer 或 processor
        tok_or_proc = None
        if load_processor:
            from transformers import AutoProcessor
            tok_or_proc = AutoProcessor.from_pretrained(local_dir, trust_remote_code=needs_remote_code)
        elif load_tokenizer:
            tok_or_proc = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=needs_remote_code)
        
        # 模型
        load_kwargs = {"trust_remote_code": needs_remote_code, "attn_implementation": "eager"}
        if "Bert" in arch or "VL" in arch or "Vision" in arch:
            model = AutoModel.from_pretrained(local_dir, **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(local_dir, **load_kwargs)
        
        elapsed = time.perf_counter() - t0
        print(f"加载模型: {elapsed:.2f}s, 参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        model.eval()
        return model, config, tok_or_proc
