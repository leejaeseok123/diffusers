import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel

sys.stdout.reconfigure(line_buffering=True)

SEED = 42
device = "cuda"

total_images = 1000
batch_size   = 10
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

H, W = 1024, 1024

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/SD3.5_clip_results.csv"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:n]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

print("[*] Loading SD3.5 Medium (1024x1024)...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
).to(device)

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

print("[*] Loading CLIP model...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("[*] CLIP 로드 완료!\n")

def calc_clip_score(images, prompts):
    inputs = clip_processor(
        text=prompts, images=images,
        return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs    = clip_model(**inputs)
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_embeds * txt_embeds).sum(dim=-1)
    return scores.cpu().float().tolist()

print("[*] Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20, height=H, width=W)
torch.cuda.synchronize()
print("[*] Warm-up 완료!\n")

os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Steps", "CLIP_Score"])

print(f"{'Steps':<8} | {'CLIP_Score':<12}")
print("-" * 25)

for T in step_sizes:
    print(f"\n[TEST] Steps={T}, Batch={batch_size}(고정)")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        all_scores = []
        with torch.inference_mode():
            for i in range(0, total_images, batch_size):
                batch_prompts = prompt_pool[i:i+batch_size]
                if not batch_prompts: break
                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                images = pipe(batch_prompts, num_inference_steps=T,
                             height=H, width=W, generator=generator).images
                all_scores.extend(calc_clip_score(images, batch_prompts))
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
