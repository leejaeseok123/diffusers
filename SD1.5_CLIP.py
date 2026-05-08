import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정
# -----------------------
SEED = 42
device = "cuda"

total_images = 1000
batch_size   = 20  # 고정 배치 (1000/20 = 50번 반복)
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

H, W = 512, 512

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/SD1.5_clip_results.csv"

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
# SD 모델 로드
# -----------------------
print("[*] Loading SD v1.5 (512x512)...")
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
# CLIP 모델 로드
# -----------------------
print("[*] Loading CLIP model...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("[*] CLIP 로드 완료!\n")

# -----------------------
# CLIP Score 계산 함수
# -----------------------
def calc_clip_score(images, prompts):
    inputs = clip_processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs    = clip_model(**inputs)
        img_embeds = outputs.image_embeds
        txt_embeds = outputs.text_embeds

        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

        scores = (img_embeds * txt_embeds).sum(dim=-1)

    return scores.cpu().float().tolist()

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
    writer.writerow(["Steps", "CLIP_Score"])

print(f"{'Steps':<8} | {'CLIP_Score':<12}")
print("-" * 25)

# -----------------------
# 실험 루프 (step만 반복)
# -----------------------
for T in step_sizes:
    print(f"\n[TEST] Steps={T}, Batch={batch_size}(고정)")

    try:
        torch.cuda.empty_cache()
        gc.collect()

        all_scores = []

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

                scores = calc_clip_score(images, batch_prompts)
                all_scores.extend(scores)

        clip_score = float(np.mean(all_scores))

        print(f"{T:<8} | {clip_score:<12.4f}")
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, clip_score])

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  → OOM! T={T} 스킵")
            torch.cuda.empty_cache()
            gc.collect()
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM"])
        else:
            print(f"  → 에러: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR"])

    finally:
        torch.cuda.empty_cache()

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
