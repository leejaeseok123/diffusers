import sys
import torch
import json
import random
import csv
import os
import gc
import time
import numpy as np
from diffusers import StableDiffusion3Pipeline

sys.stdout.reconfigure(line_buffering=True)

VERSION = "sd3.5-large-turbo"
MODEL_ID = "stabilityai/stable-diffusion-3.5-large-turbo"
H, W = 1024, 1024
BATCH_SIZE_LATENCY = 4
SEED = 42

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = f"{base_path}/results/{VERSION}_latency.csv"

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

prompt_pool = load_coco_prompts(coco_annotation_path, BATCH_SIZE_LATENCY)

print("[INFO] Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=4, guidance_scale=0.0, height=H, width=W)
torch.cuda.synchronize()
print("[INFO] Warm-up done!\n")

os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Steps", "User_Latency_sec", "BatchSize", "Resolution"])

print(f"{'Steps':<8} | {'UserLat(s)':<12}")
print("-" * 25)

for T in step_sizes:
    seed_everything(SEED)
    try:
        torch.cuda.empty_cache()
        gc.collect()

        # User Latency: 배치 4장 1번만
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            _ = pipe(
                prompt=prompt_pool[:BATCH_SIZE_LATENCY],
                num_inference_steps=T,
                guidance_scale=0.0,  # Turbo 전용
                height=H, width=W,
                generator=generator
            )
        torch.cuda.synchronize()
        user_latency = time.time() - start

        print(f"{T:<8} | {user_latency:<12.4f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, user_latency, BATCH_SIZE_LATENCY, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", BATCH_SIZE_LATENCY, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", BATCH_SIZE_LATENCY, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
