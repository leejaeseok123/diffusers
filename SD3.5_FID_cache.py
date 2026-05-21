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
# [핵심 최적화] Real 이미지의 Feature(인셉션 통계) 딱 한 번만 계산하여 캐싱
# -----------------------
print("[*] Real 이미지 인셉션 특징 추출 중 (최초 1회만 실행)...")
real_features = torch_fidelity.calculate_metrics(
    input1=real_dataset,
    cuda=True,
    fid=False,                                   # input2 없이 연산하기 위해 False로 설정
    isc=True,                                    # 내부 인셉션 모델 활성화를 위해 True 설정
    verbose=False,
    save_cpu_ram=True,
    batch_size=32,
    get_input1_features_instead_of_metrics=True  # 실제 스코어 연산 없이 인셉션 피처만 쏙 뽑아 변수에 저장
)
print("[✔] Real 이미지 특징 캐싱 완료! 이제 스텝별 계산이 극도로 빠라집니다.\n")

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
        all_images = []  # RAM에 압축된 PIL 이미지 객체 형태로 보관 (10,000장 기준 약 2.5GB 소모하여 안전함)

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
                all_images.extend(output.images)  # 디스크 파일 저장 과정을 완전히 생략

        print(f"완료 ({len(all_images)}장) → FID 계산 중...", end=" ", flush=True)
        
        # Fake용 온디맨드 데이터셋 생성
        fake_dataset = FakeMemoryDataset(all_images)

        # 캐싱된 Real 특징과 메모리 기반 Fake 데이터셋을 바인딩하여 계산
        metrics = torch_fidelity.calculate_metrics(
            input1=real_features,   # 디스크 대신 RAM에 캐싱된 Real 인셉션 통계 즉시 대입
            input2=fake_dataset,    # 디스크 I/O 없이 메모리 상에서 배치를 쪼개 통계 추출
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            batch_size=32           # 연산 속도를 높이기 위해 배치 크기 상향
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
                    input1=real_features,
                    input2=fake_dataset,
                    cuda=True, fid=True, verbose=False, save_cpu_ram=True, batch_size=32
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
        # 가비지 컬렉션을 통한 다음 스텝 메모리 확보
        all_images = []
        if 'fake_dataset' in locals(): del fake_dataset
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과: {csv_output_file}")
