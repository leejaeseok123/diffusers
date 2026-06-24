import torch
import json
import os
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusion3Pipeline

# Configuration
SEED = 42
NUM_IMAGES = 10      # 비교할 프롬프트 수
STEP = 4            # 비교할 step 수 (원하는 step으로 변경)
OUTPUT_DIR = "/home/jslee/diffusion_exper/batch_exper/fid/compare_sd35"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_coco_prompts(path, n):
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

prompts = load_coco_prompts(coco_annotation_path, NUM_IMAGES)

# SD 3.5 모델 설정
MODELS = [
    {"name": "Medium",      "id": "stabilityai/stable-diffusion-3.5-medium",      "guidance": None},
    {"name": "Large",       "id": "stabilityai/stable-diffusion-3.5-large",       "guidance": None},
    {"name": "Large Turbo", "id": "stabilityai/stable-diffusion-3.5-large-turbo", "guidance": 0.0},
]

H, W = 1024, 1024
all_images = {m["name"]: [] for m in MODELS}

for model_cfg in MODELS:
    print(f"\n[INFO] Loading {model_cfg['name']}...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_cfg["id"], torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    print(f"[INFO] Generating {NUM_IMAGES} images with T={STEP}...")
    with torch.inference_mode():
        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(device="cuda").manual_seed(SEED + idx)

            kwargs = dict(
                prompt=prompt,
                num_inference_steps=STEP,
                height=H, width=W,
                generator=generator
            )
            if model_cfg["guidance"] is not None:
                kwargs["guidance_scale"] = model_cfg["guidance"]

            output = pipe(**kwargs)
            img = output.images[0].resize((512, 512), resample=Image.LANCZOS)
            all_images[model_cfg["name"]].append(img)
            del output
            torch.cuda.empty_cache()

    del pipe
    torch.cuda.empty_cache()
    print(f"[INFO] {model_cfg['name']} done.")

# 그리드 이미지 생성
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

    try:
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_prompt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except:
        font_label = ImageFont.load_default()
        font_prompt = ImageFont.load_default()

    prompt_text = prompt if len(prompt) < 100 else prompt[:97] + "..."
    draw.text((PADDING, PADDING), f"Prompt: {prompt_text}", fill=(220, 220, 220), font=font_prompt)

    for col, ver in enumerate(versions):
        x = PADDING + col * (CELL_W + PADDING)
        y = PROMPT_H

        draw.rectangle([x, y, x + CELL_W, y + LABEL_H], fill=(50, 50, 80))
        draw.text((x + 10, y + 10), ver, fill=(255, 255, 100), font=font_label)

        img = all_images[ver][idx]
        canvas.paste(img, (x, y + LABEL_H))

    save_path = os.path.join(OUTPUT_DIR, f"compare_{idx:02d}_T{STEP}.png")
    canvas.save(save_path)
    print(f"  Saved: {save_path}")
    print(f"  Prompt: {prompt[:60]}...")

print(f"\n[SUCCESS] All comparison images saved -> {OUTPUT_DIR}")
