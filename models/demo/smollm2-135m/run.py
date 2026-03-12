"""SmolLM2 135M - 超小型语言模型 Demo
用途: 轻量级文本生成, 补全, 问答
特点: 仅135M参数, 适合边缘设备
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from demo.shared import generate_with_stream, load_model


def generate_stream(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """流式生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt")
    print(prompt, end="", flush=True)
    generate_with_stream(
        model,
        tokenizer,
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )


if __name__ == "__main__":
    model, config, tokenizer = load_model("smollm2-135m")
    
    # 示例
    prompts = [
        "The best way to learn programming is",
        "In the year 2050, artificial intelligence will",
        "def fibonacci(n):\n    '''Return the nth fibonacci number'''\n",
    ]
    
    print("\n" + "=" * 60)
    print("SmolLM2-135M 文本生成示例")
    print("=" * 60)
    
    for prompt in prompts:
        print(f"\n[Generated]")
        generate_stream(model, tokenizer, prompt, max_new_tokens=80)
        print("-" * 60)
    
    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式 (输入prompt, 按 Ctrl+C 退出)")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
            if not prompt:
                continue
            print()
            generate_stream(model, tokenizer, prompt)
        except KeyboardInterrupt:
            print("\n退出")
            break
