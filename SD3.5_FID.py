import sys
import torch
import json
import random
import csv
import os
import gc
import shutil
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
import torch_fidelity

# 출력 버퍼링 설정
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# [핵심 설정 - SD 3.5 Medium 최적화]
# -----------------------
VERSION = "v3.5_Medium"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
H, W = 1024, 1024  # SD 3.5 표준 해상도
FIXED_BATCH_SIZE = 10  # SD 3.5는 파라미터가 매우 많아 VRAM을 많이 사용하므로 4~8 권장
TOTAL_IMAGES = 10000
SEED = 42

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

# 경로 설정
base_path = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}" 
generated_root       = f"{base_path}/generated_{VERSION}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# -----------------------
# 공통 함수
# -----------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_coco_prompts(path, n):
    print(f"[*] Loading {n} COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

# -----------------------
# 1. Real 데이터셋 전처리 (1024x1024에 맞춤)
# -----------------------
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[*] {VERSION}용 Real 이미지 생성 중 ({H}x{W}, LANCZOS 필터 적용)...")
    src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
    os.makedirs(real_images_path, exist_ok=True)
    files = sorted(os.listdir(src))[:TOTAL_IMAGES]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(real_images_path, f))
        if i % 2000 == 0: print(f"  → {i}/{TOTAL_IMAGES} 완료")
    print(f"[*] {VERSION}용 Real 데이터셋 준비 완료!\n")

# -----------------------
# 2. SD 3.5 모델 로드 (핵심 변경 부분)
# -----------------------
print(f"[*] Loading {MODEL_ID}...")
# SD 3.5는 전용 Pipeline인 StableDiffusion3Pipeline을 사용해야 합니다.
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda")

# SD 3.5는 기존 enable_attention_slicing() 대신 아래 설정을 권장합니다.
pipe.enable_model_cpu_offload() # VRAM 부족 시 활성화 (성능과 메모리의 타협점)
# pipe.vae.enable_tiling()      # 고해상도 VAE 메모리 절약

pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# 3. 실험 루프
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "FID", "BatchSize", "Resolution"])

print(f"{'Steps':<8} | {'FID':<10}")
print("-" * 20)

for T in step_sizes:
    seed_everything(SEED)
    save_dir = os.path.join(generated_root, f"T{T}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[*] Steps={T}: 이미지 생성 중...", end=" ", flush=True)
        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                if not batch_prompts: break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                # SD 3.5는 height/width 인자를 직접 받습니다.
                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    height=H,
                    width=W,
                    generator=generator
                )

                for j, img in enumerate(output.images):
                    img.save(os.path.join(save_dir, f"{i+j:05d}.png"))

        # FID 계산
        print("FID 계산 중...")
        metrics = torch_fidelity.calculate_metrics(
            input1=real_images_path,
            input2=os.path.abspath(save_dir),
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True
        )
        fid = metrics['frechet_inception_distance']

        print(f"{T:<8} | {fid:<10.2f}")
        
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, fid, FIXED_BATCH_SIZE, f"{H}x{W}"])

    except Exception as e:
        print(f"\n[!] Error at T={T}: {e}")
    
    finally:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과 확인: {csv_output_file}")
