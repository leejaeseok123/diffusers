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

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VERSION = "sd3.5-large"
MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
H, W = 1024, 1024

BATCH_SIZE_LATENCY = 1  # Latency 측정용 Batch Size
TOTAL_IMAGES_LATENCY = 300  # System Latency/Throughput 측정용 이미지 수 (Batch의 배수로 설정)
SEED = 42

step_sizes = [40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = f"{base_path}/results/{VERSION}_latency_only.csv"

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Pipeline Setup
# ---------------------------------------------------------
print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES_LATENCY)

# Warm-up
print("[INFO] Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:BATCH_SIZE_LATENCY], num_inference_steps=20, height=H, width=W)
torch.cuda.synchronize()
print("[INFO] Warm-up done!\n")

# CSV Header Setup
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Steps", 
            "Batch_Size", 
            "User_Latency_sec", 
            "System_Latency_per_img_sec", 
            "Total_Time_sec", 
            "Total_Images", 
            "Resolution"
        ])

print(f"{'Steps':<8} | {'Batch':<6} | {'User Latency (s)':<18} | {'System Latency (s/img)':<22} | {'Total Time (s)':<14}")
print("-" * 80)

# ---------------------------------------------------------
# Latency Benchmark Loop
# ---------------------------------------------------------
for T in step_sizes:
    seed_everything(SEED)
    try:
        torch.cuda.empty_cache()
        gc.collect()

        # 1. User Latency (단일 배치 대기시간) 측정
        torch.cuda.synchronize()
        u_start = time.time()
        
        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            _ = pipe(
                prompt=prompt_pool[:BATCH_SIZE_LATENCY],
                num_inference_steps=T,
                height=H, width=W,
                generator=generator
            )
        
        torch.cuda.synchronize()
        user_latency = time.time() - u_start

        # 2. System Latency & Total Time (전체 Workload 처리 성능) 측정
        torch.cuda.synchronize()
        s_start = time.time()
        
        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES_LATENCY, BATCH_SIZE_LATENCY):
                batch_prompts = prompt_pool[i : i + BATCH_SIZE_LATENCY]
                if not batch_prompts:
                    break
                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                _ = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )
        
        torch.cuda.synchronize()
        total_time = time.time() - s_start
        system_latency = total_time / TOTAL_IMAGES_LATENCY

        # 결과 출력 및 저장
        print(f"{T:<8} | {BATCH_SIZE_LATENCY:<6} | {user_latency:<18.4f} | {system_latency:<22.4f} | {total_time:<14.4f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                T, 
                BATCH_SIZE_LATENCY, 
                f"{user_latency:.4f}", 
                f"{system_latency:.4f}", 
                f"{total_time:.4f}", 
                TOTAL_IMAGES_LATENCY, 
                f"{H}x{W}"
            ])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | {BATCH_SIZE_LATENCY:<6} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, BATCH_SIZE_LATENCY, "OOM", "OOM", "OOM", TOTAL_IMAGES_LATENCY, f"{H}x{W}"])
        else:
            print(f"{T:<8} | {BATCH_SIZE_LATENCY:<6} | ERROR: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, BATCH_SIZE_LATENCY, "ERROR", "ERROR", "ERROR", TOTAL_IMAGES_LATENCY, f"{H}x{W}"])
        
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Latency Benchmark finished -> {os.path.abspath(csv_output_file)}")
