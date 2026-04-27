import sys
import torch
import time
import json
import random
import csv
import os
import gc
import threading

from diffusers import StableDiffusionPipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

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
        self.thread.join()
        if not self.utils: return 0
        return sum(self.utils) / len(self.utils)

monitor = GPUUtilMonitor(handle)

# -----------------------
# 실험 설정
# -----------------------
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "sd_v15_scaling.csv"

total_images = 1000
batch_sizes = list(range(2, 65, 2))
step_sizes = [10, 20, 50]
num_runs = 2

# -----------------------
# 데이터 로드
# -----------------------
def load_coco_prompts(json_path, num_samples):
    print("[*] Loading COCO prompts...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    if len(captions) < num_samples:
        num_samples = len(captions)
    return random.sample(captions, num_samples)

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 모델 로드 (SD v1.5 - 512x512)
# -----------------------
print("[*] Loading SD v1.5 (512x512)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers 활성화 완료")
except:
    print("[*] xformers 미설치 - PyTorch SDPA 사용")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up
# -----------------------
print("[*] GPU Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20, height=512, width=512)
torch.cuda.synchronize()
print("[*] Warm-up 완료! 실험 시작합니다.\n")

# -----------------------
# CSV 초기화
# -----------------------
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Run", "Batch", "Steps", "Total_Time_s",
        "Throughput_Latency_s", "Throughput_img_s",
        "Query_Latency_s", "Peak_Mem_GB", "GPU_Util_%"
    ])

print(f"{'Run':<4} | {'Batch':<6} | {'Steps':<6} | {'Throughput_Lat':<15} | {'Throughput':<12} | {'Query_Lat':<10} | {'PeakMem':<9} | {'GPU%':<6}")
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
                # 1. Throughput 측정 (1000장 전체 처리)
                # -----------------------------------------------
                monitor.start()
                start = time.time()

                with torch.inference_mode():
                    for i in range(0, total_images, B):
                        current_batch = prompt_pool[i:i+B]
                        if not current_batch:
                            break
                        _ = pipe(current_batch, num_inference_steps=T,
                                height=512, width=512).images

                torch.cuda.synchronize()
                end = time.time()
                avg_gpu_util = monitor.stop()

                total_elapsed = end - start
                throughput_latency = total_elapsed / total_images
                throughput = total_images / total_elapsed
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                # -----------------------------------------------
                # 2. Query Latency 측정 (배치 1번만 처리)
                # -----------------------------------------------
                torch.cuda.synchronize()
                query_start = time.time()

                with torch.inference_mode():
                    _ = pipe(prompt_pool[:B], num_inference_steps=T,
                            height=512, width=512).images

                torch.cuda.synchronize()
                query_latency = time.time() - query_start

                print(f"{run:<4} | {B:<6} | {T:<6} | {throughput_latency:<15.4f} | {throughput:<12.2f} | {query_latency:<10.4f} | {peak_mem:<9.2f} | {avg_gpu_util:<6.1f}")

                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, total_elapsed, throughput_latency,
                                   throughput, query_latency, peak_mem, avg_gpu_util])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{run:<4} | {B:<6} | {T:<6} | OOM")
                    monitor.stop()
                    with open(csv_output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM", "OOM", "OOM"])
                    torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    monitor.stop()
                    raise e

print(f"\n[✔] 실험 완료! 결과: {os.path.abspath(csv_output_file)}")
