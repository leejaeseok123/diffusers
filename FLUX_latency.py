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

BATCH_SIZE = 4
TOTAL_IMAGES = 300

SEED = 42
GUIDANCE_SCALE = 3.5

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"

coco_annotation_path = (
    "/home/jslee/diffusion_exper/batch_exper/dataset/"
    "coco2014/annotation/captions_val2014.json"
)

csv_output_file = (
    f"{base_path}/results/{VERSION}_latency.csv"
)

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

    captions = sorted(
        list(set([ann["caption"] for ann in data["annotations"]]))
    )

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

prompt_pool = load_coco_prompts(
    coco_annotation_path,
    TOTAL_IMAGES,
)

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

os.makedirs(
    os.path.dirname(csv_output_file),
    exist_ok=True,
)

with open(csv_output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Steps",
        "Average_Batch_Latency(sec)",
        "Average_Image_Latency(sec)",
        "Total_Time(sec)",
        "NumImages",
        "BatchSize",
        "Resolution",
    ])

print(
    f"{'Steps':<8} | {'Batch Lat(s)':<15} | {'Image Lat(s)':<15}"
)

print("-" * 55)

# ============================================================
# Benchmark
# ============================================================

for T in step_sizes:

    seed_everything(SEED)

    try:

        torch.cuda.empty_cache()
        gc.collect()

        batch_times = []

        total_start = time.time()

        with torch.inference_mode():

            for i in range(
                0,
                TOTAL_IMAGES,
                BATCH_SIZE,
            ):

                batch_prompts = prompt_pool[
                    i : i + BATCH_SIZE
                ]

                generator = torch.Generator(
                    device="cuda"
                ).manual_seed(SEED + i)

                torch.cuda.synchronize()

                start = time.time()

                _ = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    guidance_scale=GUIDANCE_SCALE,
                    height=H,
                    width=W,
                    generator=generator,
                )

                torch.cuda.synchronize()

                batch_times.append(
                    time.time() - start
                )

        total_time = time.time() - total_start

        avg_batch_latency = np.mean(batch_times)
        avg_image_latency = total_time / TOTAL_IMAGES

        print(
            f"{T:<8} | "
            f"{avg_batch_latency:<15.4f} | "
            f"{avg_image_latency:<15.4f}"
        )

        with open(csv_output_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                T,
                avg_batch_latency,
                avg_image_latency,
                total_time,
                TOTAL_IMAGES,
                BATCH_SIZE,
                f"{H}x{W}",
            ])

    except Exception as e:

        print(f"{T:<8} | ERROR: {e}")

        with open(csv_output_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                T,
                "ERROR",
                "ERROR",
                "ERROR",
                TOTAL_IMAGES,
                BATCH_SIZE,
                f"{H}x{W}",
            ])

        torch.cuda.empty_cache()
        gc.collect()

print("\n[SUCCESS] Benchmark finished!")
print(csv_output_file)
