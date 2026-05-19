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
from diffusers import StableDiffusion3Pipeline
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
CACHE_NAME = "real_10k_1024_cache"  # 미리 생성된 캐시

step_sizes = [8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}"
generated_root       = f"{base_path}/generated_{VERSION}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID_cache.csv"

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
    save_dir = os.path.join(generated_root, f"T{T}")
    os.makedirs(save_dir, exist_ok=True)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[*] Steps={T}: 이미지 생성 중...", end=" ", flush=True)
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

                for j, img in enumerate(output.images):
                    img.save(os.path.join(save_dir, f"{i+j:05d}.png"))

        # FID 계산 (캐시 사용 → 실제 이미지 특징 추출 생략!)
        print("FID 계산 중...")
        metrics = torch_fidelity.calculate_metrics(
            input1=CACHE_NAME,                  # 캐시 사용
            input2=os.path.abspath(save_dir),
            cuda=True,
            fid=True,
            verbose=False,
            save_cpu_ram=True,
            cache_input1_name=CACHE_NAME        # 캐시 이름 명시
        )
        fid = metrics['frechet_inception_distance']

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
                with torch.inference_mode():
                    for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE // 2):
                        batch_prompts = prompt_pool[i:i+FIXED_BATCH_SIZE//2]
                        if not batch_prompts: break
                        generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                        output = pipe(
                            prompt=batch_prompts,
                            num_inference_steps=T,
                            height=H, width=W,
                            generator=generator
                        )
                        for j, img in enumerate(output.images):
                            img.save(os.path.join(save_dir, f"{i+j:05d}.png"))

                print("FID 계산 중...")
                metrics = torch_fidelity.calculate_metrics(
                    input1=CACHE_NAME,
                    input2=os.path.abspath(save_dir),
                    cuda=True, fid=True, verbose=False, save_cpu_ram=True,
                    cache_input1_name=CACHE_NAME
                )
                fid = metrics['frechet_inception_distance']
                print(f"{T:<8} | {fid:<10.2f}")
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([T, fid, FIXED_BATCH_SIZE//2, f"{H}x{W}"])

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
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] {VERSION} FID 실험 완료!")
print(f"[*] 결과: {csv_output_file}")
