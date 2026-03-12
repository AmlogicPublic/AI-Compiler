"""Gemma 2 2B IT - Google对话模型 Demo
用途: 多轮对话, 问答, 推理
特点: 指令微调版本, 支持chat模板
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run.shared import StreamingChatSession, load_model


if __name__ == "__main__":
    model, config, tokenizer = load_model("gemma-2-2b")
    chat = StreamingChatSession(model, tokenizer, default_max_new_tokens=256)
    
    # 示例对话
    print("\n" + "=" * 60)
    print("Gemma 2 2B 对话示例")
    print("=" * 60)
    
    examples = [
        "What is the capital of France?",
        "Tell me an interesting fact about it.",
    ]
    
    for q in examples:
        print(f"\n[User] {q}")
        print("[Gemma] ", end="")
        chat.chat(q, max_new_tokens=150)
    
    # 交互模式
    chat.reset()
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
            print("\nGemma> ", end="")
            chat.chat(user)
        except KeyboardInterrupt:
            print("\n退出")
            break
