import sys
import torch
import json
import random
import csv
import os
import gc
import time
import numpy as np
from diffusers import FluxPipeline

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================

VERSION = "flux1-dev"
MODEL_ID = "/mnt/ssd1/jslee/huggingface/hub/FLUX.1-dev-full"

H, W = 1024, 1024

BATCH_SIZE_LATENCY = 4
SEED = 42
GUIDANCE_SCALE = 3.5

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"

coco_annotation_path = (
    "/home/jslee/diffusion_exper/batch_exper/dataset/"
    "coco2014/annotation/captions_val2014.json"
)

csv_output_file = f"{base_path}/results/{VERSION}_latency.csv"

# ============================================================
# Utils
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, "r") as f:
        data = json.load(f)
    captions = sorted(list(set([ann["caption"] for ann in data["annotations"]])))
    return captions[:n]

# ============================================================
# Load Flux
# ============================================================

print(f"[INFO] Loading {MODEL_ID}...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

prompt_pool = load_coco_prompts(coco_annotation_path, BATCH_SIZE_LATENCY)

# ============================================================
# Warm-up
# ============================================================

print("[INFO] Warm-up...")

with torch.inference_mode():
    _ = pipe(
        prompt_pool[:2],
        num_inference_steps=20,
        guidance_scale=GUIDANCE_SCALE,
        height=H,
        width=W,
    )

torch.cuda.synchronize()
print("[INFO] Warm-up done!\n")

# ============================================================
# CSV
# ============================================================

os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

with open(csv_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Steps",
        "User_Latency_sec",
        "BatchSize",
        "Resolution",
    ])

print(f"{'Steps':<8} | {'UserLat(s)':<12}")
print("-" * 25)

# ============================================================
# Benchmark
# ============================================================

for T in step_sizes:
    seed_everything(SEED)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        # ✅ User Latency: 배치 4장 1번만 측정
        torch.cuda.synchronize()
        start = time.time()

        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            _ = pipe(
                prompt=prompt_pool[:BATCH_SIZE_LATENCY],
                num_inference_steps=T,
                guidance_scale=GUIDANCE_SCALE,
                height=H,
                width=W,
                generator=generator,
            )

        torch.cuda.synchronize()
        user_latency = time.time() - start

        print(f"{T:<8} | {user_latency:<12.4f}")

        with open(csv_output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([T, user_latency, BATCH_SIZE_LATENCY, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", BATCH_SIZE_LATENCY, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", BATCH_SIZE_LATENCY, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print("\n[SUCCESS] Benchmark finished!")
print(csv_output_file)
