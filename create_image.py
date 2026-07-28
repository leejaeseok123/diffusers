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
START_IDX = 3                       # 시작할 프롬프트 위치 (3 = 4번째 프롬프트)
NUM_PROMPTS = 3                     # 가져올 프롬프트 개수 (4, 5, 6번째)
STEPS = [10, 20, 30, 40]            # 비교할 Denoising Steps
OUTPUT_DIR = "/home/jslee/diffusion_exper/batch_exper/fid/compare_steps"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 프롬프트 로드 (시작 위치 지정 기능 추가)
def load_coco_prompts(path, start_idx, n):
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[start_idx : start_idx + n]

prompts = load_coco_prompts(coco_annotation_path, START_IDX, NUM_PROMPTS)

# 비교할 모델 설정
MODELS = [
    {"name": "v2.1", "id": "Manojb/stable-diffusion-2-1-base",           "H": 768,  "W": 768,  "dtype": torch.float16,  "type": "sd"},
    {"name": "v3.5", "id": "stabilityai/stable-diffusion-3.5-medium",    "H": 1024, "W": 1024, "dtype": torch.bfloat16, "type": "sd3"},
]

# 이미지 저장 구조
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

    # 2. 이미지 생성
    with torch.inference_mode():
        for p_idx, prompt in enumerate(prompts):
            # 파일 이름 구분을 위해 실제 프롬프트 번호 계산 (e.g., 4번째 -> real_p_num = 4)
            real_p_num = START_IDX + p_idx + 1
            
            for step in STEPS:
                print(f"[GENERATE] Model: {m_name} | Prompt #{real_p_num} | Step: {step}")
                
                # 시드 고정
                generator = torch.Generator(device="cuda").manual_seed(SEED + (START_IDX + p_idx))
                
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=step,
                    height=model_cfg["H"],
                    width=model_cfg["W"],
                    generator=generator
                )
                
                img = output.images[0].resize((512, 512), resample=Image.LANCZOS)
                results[p_idx][m_name][step] = img
                
                del output
                torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 비교 그리드 이미지 생성
# ==========================================
print("\n[INFO] Generating Step-Comparison Grids...")

CELL_W, CELL_H = 512, 512
LABEL_H = 40
PROMPT_H = 60
PADDING = 10

try:
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_prompt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except:
    font_label = ImageFont.load_default()
    font_prompt = ImageFont.load_default()

for p_idx, prompt in enumerate(prompts):
    real_p_num = START_IDX + p_idx + 1
    n_cols = len(STEPS)
    n_rows = len(MODELS)

    img_w = n_cols * (CELL_W + PADDING) + PADDING
    img_h = PROMPT_H + n_rows * (LABEL_H + CELL_H + PADDING) + PADDING

    canvas = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    prompt_text = prompt if len(prompt) < 100 else prompt[:97] + "..."
    draw.text((PADDING, PADDING), f"Prompt #{real_p_num}: {prompt_text}", fill=(220, 220, 220), font=font_prompt)

    for r_idx, model_cfg in enumerate(MODELS):
        m_name = model_cfg["name"]
        
        for c_idx, step in enumerate(STEPS):
            x = PADDING + c_idx * (CELL_W + PADDING)
            y = PROMPT_H + r_idx * (LABEL_H + CELL_H + PADDING)

            draw.rectangle([x, y, x + CELL_W, y + LABEL_H], fill=(50, 50, 80))
            draw.text((x + 10, y + 10), f"{m_name} | Step {step}", fill=(255, 255, 100), font=font_label)

            img = results[p_idx][m_name][step]
            canvas.paste(img, (x, y + LABEL_H))

    # 파일명에도 실제 프롬프트 번호(04, 05...)가 들어가도록 설정
    save_path = os.path.join(OUTPUT_DIR, f"compare_prompt_{real_p_num:02d}_steps.png")
    canvas.save(save_path)
    print(f" Saved Grid: {save_path}")

print(f"\n[SUCCESS] All step comparison grids successfully saved -> {OUTPUT_DIR}")
