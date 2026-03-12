"""Phi-3.5 Mini - 微软小型推理模型 Demo
用途: 推理, 代码生成, 数学问题
特点: 3.8B参数但推理能力强, 长上下文(128K)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run.shared import StreamingChatSession, load_model


if __name__ == "__main__":
    model, config, tokenizer = load_model("phi3.5-mini")
    chat = StreamingChatSession(model, tokenizer, default_max_new_tokens=512)
    
    # Phi擅长推理和代码
    print("\n" + "=" * 60)
    print("Phi-3.5 Mini 推理/代码示例")
    print("=" * 60)
    
    examples = [
        "Solve step by step: A train travels 120 km in 2 hours. Then it travels another 180 km in 3 hours. What is the average speed for the entire journey?",
        "Write a Python function to check if a number is prime.",
    ]
    
    for q in examples:
        print(f"\n[User]\n{q}")
        print("\n[Phi] ", end="")
        chat.chat(q, max_new_tokens=400)
        print("-" * 60)
        chat.reset()
    
    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式 (输入消息, 'reset' 清空历史, Ctrl+C 退出)")
    print("=" * 60)
    
    while True:
        try:
            user = input("\nYou> ").strip()
            if not user:
                continue
            if user.lower() == "reset":
                chat.reset()
                print("[历史已清空]")
                continue
            print("\nPhi> ", end="")
            chat.chat(user)
        except KeyboardInterrupt:
            print("\n退出")
            break
