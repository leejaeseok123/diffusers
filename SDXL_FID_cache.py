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

# 터미널 출력 실시간 확인
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 1. 설정 및 환경 준비
# -----------------------
VERSION = "SDXL_Base"
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
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# 재현성 고정 (오타 수정 완료)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# 인메모리 고속 전용 Dataset
# -----------------------
class OptimizedInferenceDataset(Dataset):
    def __init__(self, tensor_pool):
        self.tensor_pool = tensor_pool

    def __len__(self):
        return len(self.tensor_pool)

    def __getitem__(self, idx):
        return self.tensor_pool[idx]

# -----------------------
# 데이터 로드 및 공통 함수
# -----------------------
def seed_everything(target_seed):
    random.seed(target_seed)
    np.random.seed(target_seed)
    torch.manual_seed(target_seed)
    torch.cuda.manual_seed_all(target_seed)

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
# 2. 모델 로드 (타입 꼬임 정밀 튜닝)
# -----------------------
print(f"[*] Loading SDXL 1.0 ({H}x{W})...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# DDIM 스케줄러 바인딩을 먼저 수행
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 스케줄러 때문에 유입된 float32 바이어스를 파이프라인 전체 fp16 정렬로 덮어씀
pipe.to(dtype=torch.float16)

# 마지막에 안정장치로 VAE 레이어만 딱 집어서 정확하게 fp32로 업캐스팅
pipe.vae.to(dtype=torch.float32)
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
# 4. 실험 루프
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
        
        # 인메모리 파편화 방지용 단일 거대 텐서 블록 할당 (약 31.5GB)
        fake_tensor_pool = torch.zeros((TOTAL_IMAGES, 3, H, W), dtype=torch.uint8).pin_memory()

        # autocast 없이 순수 inference_mode 전개하여 정밀도 충돌 완벽 차단
        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i:i+FIXED_BATCH_SIZE]
                if not batch_prompts: break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                
                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )

                for idx, pil_img in enumerate(output.images):
                    arr = np.array(pil_img.convert('RGB'), dtype=np.uint8)
                    tensor_img = torch.from_numpy(arr).permute(2, 0, 1)
                    fake_tensor_pool[i + idx] = tensor_img
                
                if (i + FIXED_BATCH_SIZE) % 80 == 0:
                    torch.cuda.empty_cache()

        print("완료 (In-RAM 텐서화 완료) → FID 계산 중...", end=" ", flush=True)
        
        fake_dataset = OptimizedInferenceDataset(fake_tensor_pool)

        metrics = torch_fidelity.calculate_metrics(
            input1=real_images_path,
            input2=fake_dataset,
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32,
            cache_input1_name=CACHE_NAME
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
        if 'fake_tensor_pool' in locals(): del fake_tensor_pool
        if 'fake_dataset' in locals(): del fake_dataset
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과: {csv_output_file}")
