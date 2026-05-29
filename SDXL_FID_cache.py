import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from torch.utils.data import Dataset
import torch_fidelity

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정
# -----------------------
VERSION = "SDXL_Base"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 8  
TOTAL_IMAGES = 10000
SEED = 42
# torch_fidelity 내장 전용 물리 파일 캐시 이름 지정
CACHE_NAME = f"real_10k_{H}_cache_v1"

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# -----------------------
# 인메모리 고속 전용 Dataset
# -----------------------
class OptimizedInferenceDataset(Dataset):
    def __init__(self, tensor_pool):
        self.tensor_pool = tensor_pool

    def __len__(self):
        return len(self.tensor_pool)

    def __getitem__(self, idx):
        # 정제된 [3, H, W] uint8 텐서를 즉시 반환하여 변환 병목 제로화
        return self.tensor_pool[idx]

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

# -----------------------
# 모델 로드 (float16 최적화 및 VAE 정밀도 보정)
# -----------------------
print(f"[*] Loading {MODEL_ID}...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  
    use_safetensors=True
).to("cuda")

# [수정] VAE 디코더 연산만 float32로 업캐스팅하여 invalid value(NaN) 에러 원천 차단
pipe.vae.to(torch.float32)

# DDIM 스케줄러 적용
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# [캐시 최적화] 중복 연산 없는 Real 이미지 고속 캐싱
# -----------------------
print("[*] Real 이미지 인셉션 특징 추출 및 파일 캐시 등록 중 (최초 1회만)...")
_ = torch_fidelity.calculate_metrics(
    input1=real_images_path, 
    input2=real_images_path, 
    cuda=True, fid=True, verbose=False, save_cpu_ram=True, batch_size=32,
    cache_input1_name=CACHE_NAME,
    cache_input2_name=f"{CACHE_NAME}_dummy"
)
print("[✔] Real 이미지 특징 캐싱 완료! 이제 스텝별 계산 시 0초 만에 로드됩니다.\n")

# -----------------------
# 실험 루프
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

    try:
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[*] Steps={T}: 이미지 생성 중...", end=" ", flush=True)
        
        # [인메모리 최적화] 파편화 방지용 단일 거대 텐서 블록 할당 (약 31.5GB)
        fake_tensor_pool = torch.zeros((TOTAL_IMAGES, 3, H, W), dtype=torch.uint8).pin_memory()

        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i:i+FIXED_BATCH_SIZE]
                if not batch_prompts: break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                
                with torch.autocast("cuda"):
                    output = pipe(
                        prompt=batch_prompts,
                        num_inference_steps=T,
                        height=H, width=W,
                        generator=generator
                    )

                # 생성 즉시 가속 텐서 블록에 주입하여 RAM 누수 및 디스크 쓰기 병목 제거
                for idx, pil_img in enumerate(output.images):
                    arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
                    tensor_img = torch.from_numpy(arr).permute(2, 0, 1) # [3, H, W]
                    fake_tensor_pool[i + idx] = tensor_img
                
                # 주기적인 VRAM 캐시 정리
                if i % 100 == 0:
                    torch.cuda.empty_cache()

        print("완료 (In-RAM 텐서화 완료) → FID 계산 중...", end=" ", flush=True)
        
        # 인메모리 고속 데이터셋 바인딩
        fake_dataset = OptimizedInferenceDataset(fake_tensor_pool)

        metrics = torch_fidelity.calculate_metrics(
            input1=real_images_path,
            input2=fake_dataset,
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32,
            cache_input1_name=CACHE_NAME # 앞서 생성한 물리 캐시 파일 로드
        )
        fid = metrics['frechet_inception_distance']

        print(f"완료 (FID: {fid:.2f})")
        print(f"{T:<8} | {fid:<10.2f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, fid, FIXED_BATCH_SIZE, f"{H}x{W}"])

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[!] OOM at T={T} 스킵")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", FIXED_BATCH_SIZE, f"{H}x{W}"])
        else:
            print(f"\n[!] Error at T={T}: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", FIXED_BATCH_SIZE, f"{H}x{W}"])

    except Exception as e:
        print(f"\n[!] Error at T={T}: {e}")
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, "ERROR", FIXED_BATCH_SIZE, f"{H}x{W}"])

    finally:
        # 가비지 컬렉션을 통한 인메모리 풀 완전 해제
        if 'fake_tensor_pool' in locals(): del fake_tensor_pool
        if 'fake_dataset' in locals(): del fake_dataset
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과: {csv_output_file}")
