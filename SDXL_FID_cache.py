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
from diffusers import StableDiffusionXLPipeline
from torch.utils.data import Dataset
import torch_fidelity

# 터미널 출력 실시간 확인
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 1. 설정 및 환경 준비
# -----------------------
VERSION = "SDXL_Base"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 8  # 배치 사이즈 고정
TOTAL_IMAGES = 10000
SEED = 42
CACHE_NAME = f"real_10k_{H}_cache_v1"

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}"
generated_base_path  = f"{base_path}/generated_{VERSION}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# 재현성 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

# Real 데이터셋 전처리
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[*] {VERSION}용 Real 이미지 생성 중 ({H}x{W})...")
    src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
    os.makedirs(real_images_path, exist_ok=True)
    files = sorted(os.listdir(src))[:TOTAL_IMAGES]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(real_images_path, f))
    print(f"[*] Real 데이터셋 준비 완료!\n")

prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# 2. 모델 로드 및 VAE 업캐스팅 (에러 해결 핵심)
# -----------------------
print(f"[*] Loading SDXL 1.0 ({H}x{W})...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# [핵심 수정] 가중치 수동 변경 대신 Diffusers 내장 upcast_vae()를 사용하여 
# Latent 변수 타입 불일치 에러(Half vs Float)를 원천 차단합니다.
pipe.upcast_vae()

# 최신 API 반영 및 메모리 가속 활성화
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers 가속 활성화")
except ImportError:
    pipe.enable_attention_slicing()
    print("[!] xformers 없음 - Attention Slicing 사용")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# 3. Real 이미지 파일 캐시 등록
# -----------------------
print("[*] Real 이미지 인셉션 특징 추출 및 파일 캐시 등록 중 (최초 1회만)...")
_ = torch_fidelity.calculate_metrics(
    input1=real_images_path, 
    input2=real_images_path, 
    cuda=True, fid=True, verbose=False, save_cpu_ram=True, batch_size=32,
    cache_input1_name=CACHE_NAME,
    cache_input2_name=f"{CACHE_NAME}_dummy"
)
print("[✔] Real 이미지 특징 캐싱 완료!\n")

# -----------------------
# 4. 실험 루프 (완성본)
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

# 결과 저장을 위한 CSV 헤더 생성
with open(csv_output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Version", "Steps", "FID_Score"])

for steps in step_sizes:
    print(f"[*] Steps={steps}: 이미지 생성 중...")
    
    # 각 Step별 임시 생성 이미지 저장 폴더 지정
    step_img_dir = os.path.join(generated_base_path, f"steps_{steps}")
    os.makedirs(step_img_dir, exist_ok=True)
    
    # 제너레이터 시드 고정 (동일 스텝별 비교 환경 일치)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    
    try:
        # 배치 단위로 이미지 생성 진행 (10,000장 분량)
        for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
            batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
            
            with torch.inference_mode():
                outputs = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=steps,
                    height=H,
                    width=W,
                    generator=generator
                ).images
            
            # 생성된 배치 이미지 디스크 저장
            for idx, img in enumerate(outputs):
                img_idx = i + idx
                img.save(os.path.join(step_img_dir, f"gen_{img_idx:05d}.png"))
        
        print(f"[✔] Steps={steps}: 이미지 생성 완료. FID 측정 시작...")
        
        # torch_fidelity를 이용한 캐싱 기반 고속 FID 측정
        metrics = torch_fidelity.calculate_metrics(
            input1=step_img_dir,
            input2=real_images_path,
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32,
            cache_input2_name=CACHE_NAME  # 앞서 캐싱한 Real 데이터셋 활용
        )
        
        fid_score = metrics['frechet_inception_distance']
        print(f"[⭐] Steps={steps} -> FID Score: {fid_score:.4f}")
        
        # CSV 결과 누적 기록
        with open(csv_output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([VERSION, steps, fid_score])
            
    except Exception as e:
        print(f"[!] Error at T={steps}: {e}")
        
    finally:
        # 디스크 용량 확보를 위해 측정이 끝난 이미지 폴더 삭제 (필요시 주석 처리)
        if os.path.exists(step_img_dir):
            shutil.rmtree(step_img_dir)
            
        # 루프 간 GPU 메모리 완전 파편화 방지 및 비우기
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n[🎉] 모든 실험 완료! 결과가 {csv_output_file}에 저장되었습니다.")
