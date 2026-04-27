import sys
import torch
import time
import json
import random
import csv
import os
import gc
import threading
import numpy as np

from diffusers import StableDiffusionPipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 0. 재현성 고정
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# GPU 모니터링
# -----------------------
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

class GPUUtilMonitor:
    def __init__(self, handle):
        self.handle = handle
        self.utils = []
        self.stopped = False

    def start(self):
        self.utils = []
        self.stopped = False
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while not self.stopped:
            try:
                util = nvmlDeviceGetUtilizationRates(self.handle)
                self.utils.append(util.gpu)
            except:
                pass
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.thread.join()
        if not self.utils:
            return 0
        return sum(self.utils) / len(self.utils)

monitor = GPUUtilMonitor(handle)

# -----------------------
# 실험 설정
# -----------------------
device = "cuda"

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "SD_v2.1_scaling.csv"

total_images = 300
batch_sizes = list(range(2, 20, 2))
step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
num_runs = 1

# -----------------------
# 데이터 로드 (고정)
# -----------------------
def load_coco_prompts(json_path, num_samples):
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:num_samples]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 모델 로드
# -----------------------
print("[*] Loading model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
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
        "System_Latency_s",
        "User_Latency_s",
        "Throughput_img_s",
        "Peak_Mem_GB",
        "GPU_Util_%"
    ])

print("\nRun | Batch | Step | SystemLat | UserLat | Throughput | PeakMem | GPU%")
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

                # -------------------------
                # 1. SYSTEM LATENCY + THROUGHPUT
                # -------------------------
                monitor.start()
                start = time.time()

                with torch.inference_mode():
                    for i in range(0, total_images, B):
                        batch_prompts = prompt_pool[i:i+B]

                        # 🔥 매 batch마다 generator 초기화
                        generator = torch.Generator(device="cuda").manual_seed(SEED)

                        _ = pipe(
                            batch_prompts,
                            num_inference_steps=T,
                            generator=generator
                        ).images

                torch.cuda.synchronize()
                end = time.time()
                gpu_util = monitor.stop()

                total_time = end - start
                system_latency = total_time / total_images
                throughput = total_images / total_time
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                # -------------------------
                # 2. USER LATENCY (단일 요청 기준)
                # -------------------------
                torch.cuda.synchronize()
                q_start = time.time()

                generator = torch.Generator(device="cuda").manual_seed(SEED)

                with torch.inference_mode():
                    _ = pipe(
                        [prompt_pool[0]],   # 🔥 핵심 수정 (single request)
                        num_inference_steps=T,
                        generator=generator
                    ).images

                torch.cuda.synchronize()
                q_end = time.time()

                user_latency = q_end - q_start

                print(f"{run:<3} | {B:<5} | {T:<4} | {system_latency:<9.4f} | {user_latency:<8.4f} | {throughput:<10.2f} | {peak_mem:<8.2f} | {gpu_util:<6.1f}")

                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        run, B, T,
                        system_latency,
                        user_latency,
                        throughput,
                        peak_mem,
                        gpu_util
                    ])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{run:<3} | {B:<5} | {T:<4} | OOM")
                    monitor.stop()

                    with open(csv_output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM", "OOM"])

                    torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    monitor.stop()
                    raise e

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
