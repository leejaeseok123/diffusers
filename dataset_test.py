import sys
import torch
import time
import json
import random
import csv
import os
import gc

from diffusers import StableDiffusionPipeline, DDIMScheduler
from pynvml import *

# 실시간 출력
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# NVML 초기화 (GPU util)
# -----------------------
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

# -----------------------
# 설정
# -----------------------
device = "cuda"

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "batch_scaling_with_gpu_util.csv"

batch_sizes = list(range(1, 65, 2))   # 1~64 (촘촘하게)
step_sizes = [10, 20, 50]
num_runs = 2

# -----------------------
# COCO prompt 로드
# -----------------------
def load_coco_prompts(json_path, num_samples):
    print("[*] Loading COCO prompts...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    return random.sample(captions, num_samples)

prompt_pool = load_coco_prompts(coco_annotation_path, 200)

# -----------------------
# 모델 로드
# -----------------------
print("[*] Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# xformers
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers ON")
except:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up
# -----------------------
print("[*] Warm-up...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20)
torch.cuda.synchronize()

# -----------------------
# CSV 초기화
# -----------------------
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Run", "Batch", "Steps",
        "Batch_Latency_s",
        "User_Latency_s",
        "Throughput_img_s",
        "Peak_Mem_GB",
        "GPU_Util_avg_%"
    ])

print("\nRun | Batch | Step | BatchLat | UserLat | Throughput | PeakMem | GPU Util")
print("-" * 95)

# -----------------------
# 메인 실험
# -----------------------
for run in range(1, num_runs + 1):
    for T in step_sizes:
        for B in batch_sizes:

            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                prompts = prompt_pool[:B]

                gpu_utils = []

                start = time.time()

                # -----------------------
                # inference + util sampling
                # -----------------------
                with torch.inference_mode():

                    # inference 시작
                    output = pipe(
                        prompts,
                        num_inference_steps=T
                    )

                    # 실행 중 GPU util 샘플링 (짧게 반복)
                    for _ in range(10):
                        util = nvmlDeviceGetUtilizationRates(handle)
                        gpu_utils.append(util.gpu)
                        time.sleep(0.01)

                torch.cuda.synchronize()
                end = time.time()

                # -----------------------
                # metrics 계산
                # -----------------------
                batch_latency = end - start
                user_latency = batch_latency / B
                throughput = B / batch_latency
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                avg_gpu_util = sum(gpu_utils) / len(gpu_utils)

                print(f"{run:<3} | {B:<5} | {T:<4} | {batch_latency:<8.2f} | {user_latency:<7.3f} | {throughput:<10.2f} | {peak_mem:<7.2f} | {avg_gpu_util:<6.1f}")

                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        run, B, T,
                        batch_latency,
                        user_latency,
                        throughput,
                        peak_mem,
                        avg_gpu_util
                    ])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{run:<3} | {B:<5} | {T:<4} | OOM")

                    with open(csv_output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM", "OOM"])

                    torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    raise e

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
