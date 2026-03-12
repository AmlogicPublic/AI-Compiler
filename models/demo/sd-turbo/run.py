"""SD-Turbo - Stable Diffusion Turbo 图像生成 Demo
用途: 文本到图像生成 (Text-to-Image)
特点: 1-4步快速出图, 适合移动端实时生成
架构: CLIP Text Encoder + UNet + VAE Decoder (Diffusion)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import torch
from PIL import Image
from demo.shared import download_model


def load_sd_turbo():
    """加载 SD-Turbo pipeline"""
    from diffusers import StableDiffusionPipeline
    local_dir = download_model("sd-turbo")
    
    # 加载 pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        local_dir,
        torch_dtype=torch.float32,  # CPU 用 float32
        variant="fp16" if torch.cuda.is_available() else None,
    )
    
    # GPU 加速 (如果有)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("使用 CUDA")
    
    # 统计参数量
    total_params = 0
    for name in ["text_encoder", "unet", "vae"]:
        if hasattr(pipe, name):
            model = getattr(pipe, name)
            total_params += sum(p.numel() for p in model.parameters())
    print(f"加载完成, 参数量: {total_params:,}")
    
    return pipe


def generate_image(pipe, prompt, num_steps=4, guidance_scale=0.0, seed=None):
    """生成图像
    SD-Turbo 特点: 只需 1-4 步, guidance_scale=0 (不需要 CFG)
    """
    generator = torch.Generator().manual_seed(seed) if seed else None
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return image


if __name__ == "__main__":
    pipe = load_sd_turbo()
    
    print("\n" + "=" * 60)
    print("SD-Turbo 图像生成示例")
    print("=" * 60)
    
    # 示例 prompts
    prompts = [
        "a photo of a cat wearing sunglasses on a beach",
        "a futuristic city at night with neon lights",
        "a cute robot playing guitar in a park",
    ]
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}] {prompt}")
        image = generate_image(pipe, prompt, num_steps=4, seed=42+i)
        path = output_dir / f"sample_{i+1}.png"
        image.save(path)
        print(f"    保存: {path}")
    
    # 交互模式
    print("\n" + "=" * 60)
    print("交互模式:")
    print("  输入 prompt 生成图像")
    print("  输入 'seed=123' 设置随机种子")
    print("  输入 'steps=4' 设置步数 (1-4)")
    print("  Ctrl+C 退出")
    print("=" * 60)
    
    seed = None
    steps = 4
    img_count = 0
    
    while True:
        try:
            inp = input("\nPrompt> ").strip()
            if not inp:
                continue
            
            # 解析设置命令
            if inp.startswith("seed="):
                seed = int(inp.split("=")[1])
                print(f"[seed={seed}]")
                continue
            if inp.startswith("steps="):
                steps = int(inp.split("=")[1])
                print(f"[steps={steps}]")
                continue
            
            print("生成中...")
            image = generate_image(pipe, inp, num_steps=steps, seed=seed)
            img_count += 1
            path = output_dir / f"gen_{img_count}.png"
            image.save(path)
            print(f"保存: {path}")
            
            # 尝试显示图片
            try:
                image.show()
            except:
                pass
            
        except KeyboardInterrupt:
            print("\n退出")
            break
