import torch
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableDiffusion3Pipeline, DDIMScheduler

# Configuration
SEED = 42
NUM_IMAGES = 10      # 비교할 프롬프트 수
STEP = 50            # 비교할 step 수 (원하는 step으로 변경)
OUTPUT_DIR = "/home/jslee/diffusion_exper/batch_exper/fid/compare"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 프롬프트 로드
def load_coco_prompts(path, n):
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

prompts = load_coco_prompts(coco_annotation_path, NUM_IMAGES)

# 모델 설정
MODELS = [
    {"name": "v1.5",  "id": "runwayml/stable-diffusion-v1-5",              "H": 512,  "W": 512,  "dtype": torch.float16, "type": "sd"},
    {"name": "v2.1",  "id": "Manojb/stable-diffusion-2-1-base",            "H": 768,  "W": 768,  "dtype": torch.float16, "type": "sd"},
    {"name": "SDXL",  "id": "stabilityai/stable-diffusion-xl-base-1.0",    "H": 1024, "W": 1024, "dtype": torch.float16, "type": "sdxl"},
    {"name": "v3.5",  "id": "stabilityai/stable-diffusion-3.5-medium",     "H": 1024, "W": 1024, "dtype": torch.bfloat16,"type": "sd3"},
]

# 모델별 이미지 생성
all_images = {m["name"]: [] for m in MODELS}

for model_cfg in MODELS:
    print(f"\n[INFO] Loading {model_cfg['name']}...")

    # 모델 로드
    if model_cfg["type"] == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_cfg["id"], torch_dtype=model_cfg["dtype"], safety_checker=None
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif model_cfg["type"] == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_cfg["id"], torch_dtype=model_cfg["dtype"], use_safetensors=True
        ).to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    elif model_cfg["type"] == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_cfg["id"], torch_dtype=model_cfg["dtype"]
        ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    print(f"[INFO] Generating {NUM_IMAGES} images with T={STEP}...")
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(device="cuda").manual_seed(SEED + idx)
            output = pipe(
                prompt=prompt,
                num_inference_steps=STEP,
                height=model_cfg["H"],
                width=model_cfg["W"],
                generator=generator
            )
            # 512x512로 통일해서 저장
            img = output.images[0].resize((512, 512), resample=Image.LANCZOS)
            all_images[model_cfg["name"]].append(img)
            del output
            torch.cuda.empty_cache()

    # 메모리 해제
    del pipe
    torch.cuda.empty_cache()
    print(f"[INFO] {model_cfg['name']} done.")

# 그리드 이미지 생성 (프롬프트별로 각 모델 비교)
print("\n[INFO] Generating comparison grids...")

CELL_W, CELL_H = 512, 512
LABEL_H = 40
PROMPT_H = 60
PADDING = 10
versions = [m["name"] for m in MODELS]

for idx, prompt in enumerate(prompts):
    n_cols = len(versions)
    img_w = n_cols * (CELL_W + PADDING) + PADDING
    img_h = PROMPT_H + LABEL_H + CELL_H + PADDING * 2

    canvas = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # 프롬프트 텍스트
    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_prompt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font_label = ImageFont.load_default()
        font_prompt = ImageFont.load_default()

    # 프롬프트 표시 (길면 자르기)
    prompt_text = prompt if len(prompt) < 100 else prompt[:97] + "..."
    draw.text((PADDING, PADDING), f"Prompt: {prompt_text}", fill=(220, 220, 220), font=font_prompt)

    # 각 모델 이미지 + 라벨
    for col, ver in enumerate(versions):
        x = PADDING + col * (CELL_W + PADDING)
        y = PROMPT_H

        # 모델 이름 라벨
        draw.rectangle([x, y, x + CELL_W, y + LABEL_H], fill=(50, 50, 80))
        draw.text((x + 10, y + 10), ver, fill=(255, 255, 100), font=font_label)

        # 이미지
        img = all_images[ver][idx]
        canvas.paste(img, (x, y + LABEL_H))

    # 저장
    save_path = os.path.join(OUTPUT_DIR, f"compare_{idx:02d}_T{STEP}.png")
    canvas.save(save_path)
    print(f"  Saved: {save_path}")
    print(f"  Prompt: {prompt[:60]}...")

print(f"\n[SUCCESS] All comparison images saved -> {OUTPUT_DIR}")
