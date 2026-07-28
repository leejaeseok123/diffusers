import torch
import json
import os
import gc
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, DDIMScheduler

# ==========================================
# Configuration
# ==========================================
SEED = 42
NUM_PROMPTS = 3                     # 비교할 프롬프트 수 (3개)
STEPS = [10, 20, 30, 40]            # 비교할 Denoising Steps (40 추가)
OUTPUT_DIR = "/home/jslee/diffusion_exper/batch_exper/fid/compare_steps"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 프롬프트 로드
def load_coco_prompts(path, n):
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

prompts = load_coco_prompts(coco_annotation_path, NUM_PROMPTS)

# 비교할 모델 설정
MODELS = [
    {"name": "v2.1", "id": "Manojb/stable-diffusion-2-1-base",           "H": 768,  "W": 768,  "dtype": torch.float16,  "type": "sd"},
    {"name": "v3.5", "id": "stabilityai/stable-diffusion-3.5-medium",    "H": 1024, "W": 1024, "dtype": torch.bfloat16, "type": "sd3"},
]

# 이미지 저장 구조: results[prompt_idx][model_name][step] = PIL.Image
results = {p_idx: {m["name"]: {} for m in MODELS} for p_idx in range(NUM_PROMPTS)}

# ==========================================
# 모델별 / 스텝별 이미지 생성 Loop
# ==========================================
for model_cfg in MODELS:
    m_name = model_cfg["name"]
    print(f"\n[INFO] Loading Model: {m_name} ({model_cfg['id']})...")

    # 1. 모델 로드
    if model_cfg["type"] == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_cfg["id"], torch_dtype=model_cfg["dtype"], safety_checker=None
        ).to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif model_cfg["type"] == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_cfg["id"], torch_dtype=model_cfg["dtype"]
        ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    # 2. 이미지 생성 (프롬프트 -> 스텝 순으로 반복)
    with torch.inference_mode():
        for p_idx, prompt in enumerate(prompts):
            for step in STEPS:
                print(f"[GENERATE] Model: {m_name} | Prompt #{p_idx+1} | Step: {step}")
                
                # 시드 고정 (동일 프롬프트/스텝에 대해 시드 일치)
                generator = torch.Generator(device="cuda").manual_seed(SEED + p_idx)
                
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=step,
                    height=model_cfg["H"],
                    width=model_cfg["W"],
                    generator=generator
                )
                
                # 512x512 해상도로 통일하여 리사이즈 후 저장
                img = output.images[0].resize((512, 512), resample=Image.LANCZOS)
                results[p_idx][m_name][step] = img
                
                del output
                torch.cuda.empty_cache()

    # 메모리 해제
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[INFO] {m_name} Pipeline Cleared.")

# ==========================================
# 비교 그리드 이미지 생성 (2행 4열 구조)
# ==========================================
print("\n[INFO] Generating Step-Comparison Grids...")

CELL_W, CELL_H = 512, 512
LABEL_H = 40
PROMPT_H = 60
PADDING = 10

# 폰트 설정
try:
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_prompt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except:
    font_label = ImageFont.load_default()
    font_prompt = ImageFont.load_default()

for p_idx, prompt in enumerate(prompts):
    n_cols = len(STEPS)  # 4 (Steps: 10, 20, 30, 40)
    n_rows = len(MODELS) # 2 (v2.1, v3.5)

    img_w = n_cols * (CELL_W + PADDING) + PADDING
    img_h = PROMPT_H + n_rows * (LABEL_H + CELL_H + PADDING) + PADDING

    canvas = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # 1. 상단 프롬프트 출력
    prompt_text = prompt if len(prompt) < 100 else prompt[:97] + "..."
    draw.text((PADDING, PADDING), f"Prompt #{p_idx+1}: {prompt_text}", fill=(220, 220, 220), font=font_prompt)

    # 2. 그리드 배치 (행: 모델 / 열: 스텝)
    for r_idx, model_cfg in enumerate(MODELS):
        m_name = model_cfg["name"]
        
        for c_idx, step in enumerate(STEPS):
            x = PADDING + c_idx * (CELL_W + PADDING)
            y = PROMPT_H + r_idx * (LABEL_H + CELL_H + PADDING)

            # 상단 라벨 (예: "v2.1 | Step: 40")
            draw.rectangle([x, y, x + CELL_W, y + LABEL_H], fill=(50, 50, 80))
            draw.text((x + 10, y + 10), f"{m_name} | Step {step}", fill=(255, 255, 100), font=font_label)

            # 이미지 붙이기
            img = results[p_idx][m_name][step]
            canvas.paste(img, (x, y + LABEL_H))

    # 저장
    save_path = os.path.join(OUTPUT_DIR, f"compare_prompt_{p_idx:02d}_steps.png")
    canvas.save(save_path)
    print(f" Saved Grid: {save_path}")

print(f"\n[SUCCESS] All step comparison grids successfully saved -> {OUTPUT_DIR}")
