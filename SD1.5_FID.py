import sys
import torch
import json
import random
import csv
import os
import gc
import shutil
import numpy as np

from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch_fidelity

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정
# -----------------------
SEED = 42
device = "cuda"

total_images = 10000
# 실험할 배치 사이즈와 스텝 수
batch_sizes  = [1, 2, 4, 8, 16, 32, 64, 80, 96, 128]
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

H, W = 512, 512

# 경로 설정 (끝에 /를 붙여 디렉토리임을 명시)
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = os.path.abspath("/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k") + "/"
generated_root       = "/home/jslee/diffusion_exper/batch_exper/fid/generated"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/fid_results.csv"

# -----------------------
# Seed 고정
# -----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# COCO prompt 로드
# -----------------------
def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:n]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 모델 로드
# -----------------------
print("[*] Loading SD v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers ON")
except ImportError:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up (GPU 엔진 가동)
# -----------------------
print("[*] Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=5, height=H, width=W)
torch.cuda.synchronize()
print("[*] Warm-up 완료!\n")

# -----------------------
# CSV 초기화
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Batch", "Steps", "FID"])

print(f"{'Batch':<8} | {'Steps':<8} | {'FID':<10}")
print("-" * 35)

# -----------------------
# 실험 루프
# -----------------------
for T in step_sizes:
    for B in batch_sizes:
        print(f"\n[TEST] Steps={T}, Batch={B}")

        save_dir = os.path.join(generated_root, f"T{T}_B{B}")
        os.makedirs(save_dir, exist_ok=True)
        abs_save_dir = os.path.abspath(save_dir) + "/"

        try:
            # 1. 이미지 생성 단계
            existing = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
            if existing >= total_images:
                print(f"  → 이미 생성됨 ({existing}장) 스킵")
            else:
                torch.cuda.empty_cache()
                gc.collect()

                with torch.inference_mode():
                    for i in range(0, total_images, B):
                        batch_prompts = prompt_pool[i:i+B]
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

                print(f"  → {total_images}장 생성 완료")

            # 2. FID 계산 단계
            print(f"  → FID 계산 중...")
            metrics = torch_fidelity.calculate_metrics(
                input1=real_images_path,
                input2=abs_save_dir,
                cuda=True,
                fid=True,
                verbose=False,
                save_cpu_ram=True # 메모리 절약
            )
            fid = metrics['frechet_inception_distance']

            print(f"{B:<8} | {T:<8} | {fid:<10.2f}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([B, T, fid])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  → OOM 발생! B={B}, T={T} 스킵")
                torch.cuda.empty_cache()
                gc.collect()
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([B, T, "OOM"])
            else:
                print(f"  → 런타임 에러: {e}")
                continue # 다음 배치로 진행
        
        except Exception as e:
            print(f"  → 예상치 못한 에러: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([B, T, f"ERROR: {type(e).__name__}"])

        finally:
            # 다음 실험을 위해 이미지 삭제 및 메모리 정리
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                print(f"  → 이미지 삭제 완료")
            torch.cuda.empty_cache()

print(f"\n[✔] 실험 완료 → {os.path.abspath(csv_output_file)}")
