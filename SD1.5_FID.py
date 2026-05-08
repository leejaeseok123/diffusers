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
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch_fidelity

# 출력 버퍼링 설정
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# [핵심 설정]
# -----------------------
MODEL_ID = "runwayml/stable-diffusion-v1-5"
H, W = 512, 512  # SD v1.5 표준 해상도
FIXED_BATCH_SIZE = 100  # VRAM 상황에 따라 16~64 조절 가능
TOTAL_IMAGES = 10000
SEED = 42

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

# 경로 설정 (사용자 환경에 맞게 수정됨)
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"/home/jslee/diffusion_exper/batch_exper/fid/real_10k_{H}"
generated_root       = "/home/jslee/diffusion_exper/batch_exper/fid/generated"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/v1.5_FID.csv"

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
    # 중복 제거 및 정렬로 일관성 유지
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

# -----------------------
# 1. Real 데이터셋 전처리 (FID 수치 정상화의 핵심)
# -----------------------
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[*] Real 이미지 생성 중 ({H}x{W}, LANCZOS 필터 적용)...")
    src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
    os.makedirs(real_images_path, exist_ok=True)
    files = sorted(os.listdir(src))[:TOTAL_IMAGES]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        # LANCZOS 필터를 써야 생성 이미지와 통계적 특성(FID)이 맞습니다.
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(real_images_path, f))
        if i % 2000 == 0: print(f"  → {i}/{TOTAL_IMAGES} 완료")
    print("[*] Real 데이터셋 준비 완료!\n")

# -----------------------
# 2. SD v1.5 모델 로드
# -----------------------
print(f"[*] Loading {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# DDIM 스케줄러 설정
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers 가속 활성화")
except ImportError:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# 3. 실험 루프
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Steps", "FID", "BatchSize", "Resolution"])

print(f"{'Steps':<8} | {'FID':<10}")
print("-" * 20)

for T in step_sizes:
    seed_everything(SEED) # 매 실험마다 동일 시드 시작
    save_dir = os.path.join(generated_root, f"v15_T{T}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()

        # 이미지 생성 루프
        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                if not batch_prompts: break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                output = pipe(
                    batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )

                for j, img in enumerate(output.images):
                    img.save(os.path.join(save_dir, f"{i+j:05d}.png"))

        # FID 계산
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
        # 생성된 이미지 삭제 (디스크 용량 확보)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

print(f"\n[✔] v1.5 실험 완료! 결과 저장: {csv_output_file}")
