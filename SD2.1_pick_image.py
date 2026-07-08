import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import AutoProcessor, AutoModel
import ImageReward as RM

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정 (SD2.1)
# -----------------------
SEED = 42
device = "cuda"

total_images = 1000
batch_size   = 1
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

H, W = 512, 512

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/SD2.1_pick_image.csv"

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
# SD2.1 모델 로드
# -----------------------
print("[*] Loading SD2.1 (stabilityai/stable-diffusion-2-1)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
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
# PickScore 모델 로드 (CLIP-H 기반)
# -----------------------
print("[*] Loading PickScore...")
pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pick_model     = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()

# -----------------------
# ImageReward 모델 로드 (로컬 경로 지정)
# -----------------------
print("[*] Loading ImageReward from local path...")
local_weights_dir = "/home/jslee/diffusion_exper/weights/ImageReward"

# download_root에 파일이 있으면 다운로드를 건너뛰고 해당 파일들을 로드합니다.
reward_model = RM.load("ImageReward-v1.0", download_root=local_weights_dir).to(device)
print("[*] 평가 모델 로드 완료!\n")

# -----------------------
# PickScore 계산 함수 (배치)
# -----------------------
def calc_pickscore(images, prompts):
    image_inputs = pick_processor(images=images, return_tensors="pt", padding=True).to(device)
    text_inputs  = pick_processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        image_embs = pick_model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)
        text_embs  = pick_model.get_text_features(**text_inputs)
        text_embs  = text_embs / text_embs.norm(dim=-1, keepdim=True)
        scale = pick_model.logit_scale.exp()
        scores = scale * (text_embs * image_embs).sum(dim=-1)
    return scores.cpu().float().tolist()

# -----------------------
# ImageReward 계산 함수 (1장씩, 배치 API 없음)
# -----------------------
def calc_imagereward(images, prompts):
    scores = []
    for img, prompt in zip(images, prompts):
        with torch.no_grad():
            s = reward_model.score(prompt, img)
        scores.append(float(s))
    return scores

# -----------------------
# Warm-up
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
    writer.writerow(["Steps", "PickScore", "ImageReward"])

print(f"{'Steps':<8} | {'PickScore':<12} | {'ImageReward':<12}")
print("-" * 40)

# -----------------------
# 실험 루프 (step만 반복)
# -----------------------
for T in step_sizes:
    print(f"\n[TEST] Steps={T}, Batch={batch_size}(고정)")

    try:
        torch.cuda.empty_cache()
        gc.collect()

        pick_scores, ir_scores = [], []

        with torch.inference_mode():
            for i in range(0, total_images, batch_size):
                batch_prompts = prompt_pool[i:i+batch_size]
                if not batch_prompts:
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                images = pipe(
                    batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                ).images

                pick_scores.extend(calc_pickscore(images, batch_prompts))
                ir_scores.extend(calc_imagereward(images, batch_prompts))

        pick_mean = float(np.mean(pick_scores))
        ir_mean   = float(np.mean(ir_scores))

        print(f"{T:<8} | {pick_mean:<12.4f} | {ir_mean:<12.4f}")
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, pick_mean, ir_mean])

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  → OOM! T={T} 스킵")
            torch.cuda.empty_cache()
            gc.collect()
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", "OOM"])
        else:
            print(f"  → 에러: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", "ERROR"])

    finally:
        torch.cuda.empty_cache()

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
