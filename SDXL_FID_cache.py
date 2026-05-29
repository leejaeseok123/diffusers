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
VERSION = "SDXL_Base_FP32"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 8  
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
# 2. 모델 로드 (완전한 float32 통일)
# -----------------------
print(f"[*] Loading SDXL 1.0 in Full FP32 ({H}x{W})...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,     # [핵심] fp16 대신 완전히 fp32로 로드
    use_safetensors=True
    # variant="fp16" 라인 제거 (fp32 순정 가중치 다운로드)
).to("cuda")

# 메모리 최적화 켜기 (fp32는 무겁기 때문에 필수입니다)
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
# 4. 실험 루프 (순정 코드로 원상복구)
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

with open(csv_output_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Version", "Steps", "FID_Score"])

for steps in step_sizes:
    print(f"[*] Steps={steps}: 이미지 생성 중...")
    
    step_img_dir = os.path.join(generated_base_path, f"steps_{steps}")
    os.makedirs(step_img_dir, exist_ok=True)
    
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    
    try:
        for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
            batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
            
            # 모든 텐서가 fp32이므로 복잡한 분리 처리나 autocast 없이 바로 호출 가능
            with torch.inference_mode():
                outputs = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=steps,
                    height=H,
                    width=W,
                    generator=generator
                ).images
            
            # 이미지 저장
            for idx, img in enumerate(outputs):
                img_idx = i + idx
                img.save(os.path.join(step_img_dir, f"gen_{img_idx:05d}.png"))
        
        print(f"[✔] Steps={steps}: 이미지 생성 완료. FID 측정 시작...")
        
        # FID 측정
        metrics = torch_fidelity.calculate_metrics(
            input1=step_img_dir,
            input2=real_images_path,
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32,
            cache_input2_name=CACHE_NAME
        )
        
        fid_score = metrics['frechet_inception_distance']
        print(f"[⭐] Steps={steps} -> FID Score: {fid_score:.4f}")
        
        with open(csv_output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([VERSION, steps, fid_score])
            
    except Exception as e:
        print(f"[!] Error at T={steps}: {e}")
        
    finally:
        if os.path.exists(step_img_dir):
            shutil.rmtree(step_img_dir)
            
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n[🎉] 모든 실험 완료! 결과가 {csv_output_file}에 저장되었습니다.")
