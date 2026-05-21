import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import Dataset
import torch_fidelity

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정
# -----------------------
VERSION = "v3.5_Medium"
MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 4
TOTAL_IMAGES = 10000
SEED = 42
# torch_fidelity 내장 시스템이 로컬 디렉토리에 물리적으로 보관할 캐시 이름 지정
CACHE_NAME = f"real_10k_{H}_cache_v1"

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# -----------------------
# torch_fidelity 메모리 입력을 위한 온디맨드 Dataset
# -----------------------
class RealMemoryDataset(Dataset):
    def __init__(self, path, n):
        self.files = sorted([
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])[:n]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W] uint8 텐서

class FakeMemoryDataset(Dataset):
    def __init__(self, pil_images_list):
        self.images = pil_images_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # torch_fidelity가 특정 인덱스를 요구할 때만 즉석에서 텐서 변환 (메모리 절약)
        arr = np.array(self.images[idx].convert('RGB'), dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W] uint8 텐서

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
# Real 데이터셋 전처리 및 데이터셋 객체 준비
# -----------------------
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[*] {VERSION}용 Real 이미지 생성 중 ({H}x{W})...")
    src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
    os.makedirs(real_images_path, exist_ok=True)
    files = sorted(os.listdir(src))[:TOTAL_IMAGES]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(real_images_path, f))
        if i % 2000 == 0: print(f"  → {i}/{TOTAL_IMAGES} 완료")
    print(f"[*] Real 데이터셋 준비 완료!\n")

real_dataset = RealMemoryDataset(real_images_path, TOTAL_IMAGES)

# -----------------------
# 모델 로드 (bfloat16 - SD3.5 공식 권장)
# -----------------------
print(f"[*] Loading {MODEL_ID}...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# [핵심 변경 1] 하드웨어 파일 기반의 영구 캐시 생성 단계
# -----------------------
print("[*] Real 이미지 인셉션 특징 추출 및 파일 캐시 등록 중 (최초 1회만 연산)...")
_ = torch_fidelity.calculate_metrics(
    input1=real_dataset,               # 명확한 Dataset 객체를 대입하여 에러 차단
    input2=real_dataset,               # 바이너리 제약 조건 우회를 위해 임시로 복사 대입
    cuda=True,
    fid=True,
    verbose=False,
    save_cpu_ram=True,
    batch_size=32,
    cache_input1_name=CACHE_NAME,      # 이 이름으로 ~/.cache/torch/fidelity에 .pth 통계 파일이 자동 저장됨
    cache_input2_name=f"{CACHE_NAME}_temp"
)
print("[✔] Real 이미지 특징 캐싱 완료! 이제 스텝별 계산 시 디스크를 전혀 읽지 않고 0초만에 통과합니다.\n")

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
        all_images = []  # RAM에 압축된 PIL 이미지 객체 형태로 보관 (10,000장 기준 약 2.5GB 소모하여 매우 안전)

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
                all_images.extend(output.images)  # 디스크에 파일을 쓰지 않아 쓰기 병목 소멸

        print(f"완료 ({len(all_images)}장) → FID 계산 중...", end=" ", flush=True)
        
        # Fake용 온디맨드 데이터셋 생성
        fake_dataset = FakeMemoryDataset(all_images)

        # [핵심 변경 2] 캐시 이름을 명시하여 기존 캐시 데이터 강제 로드
        metrics = torch_fidelity.calculate_metrics(
            input1=real_dataset,            # 규격 매칭을 위해 객체를 주되, 실제 연산은 일어나지 않음
            input2=fake_dataset,            # 디스크 I/O 없이 메모리 상에서 배치를 쪼개 통계 추출
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32,
            cache_input1_name=CACHE_NAME    # 캐시 이름이 일치하므로 물리 파일 로딩 후 즉시 계산 돌입
        )
        fid = metrics['frechet_inception_distance']

        print(f"완료 (FID: {fid:.2f})")
        print(f"{T:<8} | {fid:<10.2f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, fid, FIXED_BATCH_SIZE, f"{H}x{W}"])

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[!] OOM at T={T} → 배치 줄여서 재시도...")
            torch.cuda.empty_cache()
            gc.collect()

            try:
                all_images = []
                HALF_BATCH = FIXED_BATCH_SIZE // 2
                with torch.inference_mode():
                    for i in range(0, TOTAL_IMAGES, HALF_BATCH):
                        batch_prompts = prompt_pool[i:i+HALF_BATCH]
                        if not batch_prompts: break
                        generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                        output = pipe(
                            prompt=batch_prompts,
                            num_inference_steps=T,
                            height=H, width=W,
                            generator=generator
                        )
                        all_images.extend(output.images)

                fake_dataset = FakeMemoryDataset(all_images)
                metrics = torch_fidelity.calculate_metrics(
                    input1=real_dataset,
                    input2=fake_dataset,
                    cuda=True, fid=True, verbose=False, save_cpu_ram=True, batch_size=32,
                    cache_input1_name=CACHE_NAME
                )
                fid = metrics['frechet_inception_distance']
                print(f"{T:<8} | {fid:<10.2f}")
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([T, fid, HALF_BATCH, f"{H}x{W}"])

            except Exception as e2:
                print(f"\n[!] 재시도 실패 T={T}: {e2}")
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

    finally:
        # 가비지 컬렉션을 통한 메모리 완전 해제
        all_images = []
        if 'fake_dataset' in locals(): del fake_dataset
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과: {csv_output_file}")
