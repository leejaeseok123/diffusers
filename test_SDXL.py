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

from diffusers import StableDiffusionXLPipeline
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 재현성 고정
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
            except: pass
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        if hasattr(self, "thread"):
            self.thread.join()
        if not self.utils: return 0
        return sum(self.utils) / len(self.utils)

monitor = GPUUtilMonitor(handle)

# -----------------------
# 실험 설정
# -----------------------
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "SDXL_scaling.csv"

total_images = 300
batch_sizes  = [1, 2, 4, 8, 16, 32, 64, 80, 96, 128]
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
num_runs     = 1
H, W         = 1024, 1024

# -----------------------
# 데이터 로드
# -----------------------
def load_coco_prompts(json_path, num_samples):
    print("[*] Loading COCO prompts...")
    if not os.path.exists(json_path):
        print(f"[!] 경로 없음: {json_path}")
        return ["a photo of a cat"] * num_samples
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = sorted(list(set([ann["caption"] for ann in data["annotations"]])))
    return captions[:num_samples]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 모델 로드
# -----------------------
print("[*] Loading SDXL 1.0...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to(device)

# VAE 타입 충돌 방지 (공식 권장)
pipe.upcast_vae()

pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers ON")
except ImportError:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up
# -----------------------
print("[*] Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=10, height=H, width=W).images
torch.cuda.synchronize()
print("[*] Warm-up 완료\n")

# -----------------------
# CSV 초기화
# -----------------------
with open(csv_output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Run", "Batch", "Steps",
        "System_Latency_s", "User_Latency_s",
        "Throughput_img_s", "Peak_Mem_GB", "GPU_Util_%"
    ])

print(f"{'Run':<4} | {'Batch':<6} | {'Steps':<6} | {'SystemLat':<11} | {'UserLat':<11} | {'Throughput':<12} | {'PeakMem':<10} | {'GPU%':<6}")
print("-" * 100)

# -----------------------
# 메인 실험 루프
# -----------------------
for run in range(1, num_runs + 1):
    for T in step_sizes:
        for B in batch_sizes:
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                # -----------------------------------------------
                # 1. System Latency + Throughput
                # -----------------------------------------------
                monitor.start()
                start = time.time()

                with torch.inference_mode():
                    for i in range(0, total_images, B):
                        batch_prompts = prompt_pool[i:i+B]
                        if not batch_prompts: break
                        generator = torch.Generator(device="cuda").manual_seed(SEED)
                        _ = pipe(
                            batch_prompts,
                            num_inference_steps=T,
                            height=H, width=W,
                            generator=generator
                        ).images  # VAE 디코딩 포함

                torch.cuda.synchronize()
                end = time.time()
                gpu_util = monitor.stop()

                total_time     = end - start
                system_latency = total_time / total_images
                throughput     = total_images / total_time
                peak_mem       = torch.cuda.max_memory_allocated() / 1024**3

                # -----------------------------------------------
                # 2. User Latency (배치 1번 처리)
                # -----------------------------------------------
                torch.cuda.synchronize()
                u_start = time.time()

                with torch.inference_mode():
                    generator = torch.Generator(device="cuda").manual_seed(SEED)
                    _ = pipe(
                        prompt_pool[:B],
                        num_inference_steps=T,
                        height=H, width=W,
                        generator=generator
                    ).images  # VAE 디코딩 포함

                torch.cuda.synchronize()
                user_latency = time.time() - u_start

                print(f"{run:<4} | {B:<6} | {T:<6} | {system_latency:<11.4f} | {user_latency:<11.4f} | {throughput:<12.2f} | {peak_mem:<10.2f} | {gpu_util:<6.1f}")

                with open(csv_output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, system_latency, user_latency, throughput, peak_mem, gpu_util])

            except RuntimeError as e:
                monitor.stop()
                if "out of memory" in str(e).lower():
                    print(f"{run:<4} | {B:<6} | {T:<6} | OOM")
                    with open(csv_output_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM", "OOM"])
                    torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    raise e

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
