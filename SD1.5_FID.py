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
batch_sizes  = [4, 8, 12, 16, 20, 30, 60, 100, 120]
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

H, W = 512, 512

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
generated_root       = "/home/jslee/diffusion_exper/batch_exper/fid/generated"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/fid_results.csv"

# -----------------------
# Seed
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
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers ON")
except:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up
# -----------------------
print("[*] Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20, height=H, width=W)
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

        try:
            # 이미 생성된 이미지 있으면 스킵
            existing = len([f for f in os.listdir(save_dir) if f.endswith('.png')])
            if existing >= total_images:
                print(f"  → 이미 생성됨 ({existing}장) 스킵")
            else:
                torch.cuda.empty_cache()
                gc.collect()

                # 이미지 생성 + 저장
                with torch.inference_mode():
                    for i in range(0, total_images, B):
                        batch_prompts = prompt_pool[i:i+B]
                        if not batch_prompts:
                            break

                        generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                        images = pipe(
                            batch_prompts,
                            num_inference_steps=T,
                            height=H, width=W,
                            generator=generator
                        ).images

                        for j, img in enumerate(images):
                            img.save(f"{save_dir}/{i+j:05d}.png")

                print(f"  → {total_images}장 생성 완료")

            # FID 계산
            print(f"  → FID 계산 중...")
            metrics = torch_fidelity.calculate_metrics(
                input1=real_images_path,
                input2=save_dir,
                cuda=True,
                fid=True,
                verbose=False
            )
            fid = metrics['frechet_inception_distance']

            print(f"{B:<8} | {T:<8} | {fid:<10.2f}")

            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([B, T, fid])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  → OOM! B={B}, T={T} 스킵")
                torch.cuda.empty_cache()
                gc.collect()
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([B, T, "OOM"])
            else:
                raise e

        finally:
            # 항상 이미지 삭제 (OOM 나도 삭제)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                print(f"  → 이미지 삭제 완료")

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
