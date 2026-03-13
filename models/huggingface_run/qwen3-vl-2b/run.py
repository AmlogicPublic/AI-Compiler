"""Qwen3-VL 2B - 阿里多模态视觉语言模型 Demo
用途: 图片描述, 视觉问答, OCR
特点: 支持图片和视频输入
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
from PIL import Image
from huggingface_run.shared import download_model, generate_with_stream


def load_qwen_vl():
    """加载 Qwen-VL 模型和 processor"""
    import warnings
    from transformers import AutoProcessor
    from transformers import AutoModelForImageTextToText
    
    local_dir = download_model("qwen3-vl-2b")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            local_dir, 
            trust_remote_code=True,
            attn_implementation="eager"
        )
    
    model.eval()
    print(f"加载完成, 参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


def describe_image_stream(model, processor, image_path, question="Describe this image in detail."):
    """流式描述图片"""
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    
    return generate_with_stream(
        model,
        processor.tokenizer,
        inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )


def create_demo_image(path):
    """创建示例图片"""
    img = Image.new('RGB', (200, 200), color='skyblue')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill='yellow', outline='orange')
    draw.rectangle([80, 160, 120, 200], fill='brown')
    draw.text((60, 10), "Demo", fill='black')
    img.save(path)
    return path


if __name__ == "__main__":
    model, processor = load_qwen_vl()
    
    print("\n" + "=" * 60)
    print("Qwen3-VL 多模态示例")
    print("=" * 60)
    
    # 创建或使用示例图片 (在 run 目录内)
    demo_dir = Path(__file__).parent
    demo_img = str(demo_dir / "demo_image.png")
    if not os.path.exists(demo_img):
        create_demo_image(demo_img)
        print(f"\n创建示例图片: {demo_img}")
    
    # 描述图片
    print(f"\n[图片] {demo_img}")
    print(f"[问题] Describe this image.")
    print("[回答] ", end="")
    describe_image_stream(model, processor, demo_img, "Describe this image.")
    
    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式:")
    print("  输入图片路径和问题, 格式: 图片路径 | 问题")
    print("  例: cat.jpg | What animal is this?")
    print("  仅输入问题则继续使用当前图片")
    print("  Ctrl+C 退出")
    print("=" * 60)
    
    current_image = demo_img
    
    while True:
        try:
            inp = input("\n> ").strip()
            if not inp:
                continue
            
            if "|" in inp:
                parts = inp.split("|", 1)
                img_path = parts[0].strip()
                question = parts[1].strip()
                assert os.path.exists(img_path), f"图片不存在: {img_path}"
                current_image = img_path
            else:
                question = inp
            
            print(f"[图片] {current_image}")
            print(f"[问题] {question}")
            print("[回答] ", end="")
            describe_image_stream(model, processor, current_image, question)
            
        except KeyboardInterrupt:
            print("\n退出")
            break
