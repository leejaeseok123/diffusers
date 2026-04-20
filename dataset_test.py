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

# 실시간 출력 설정 (터미널에서 즉시 확인 가능)
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# GPU 모니터링 클래스 (비동기)
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
            time.sleep(0.01) # 10ms 단위

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
csv_output_file = "rtx6000_optimized_scaling.csv"

# RTX 6000의 성능을 보기 위해 배치를 촘촘하게 구성
batch_sizes = list(range(1, 65, 2)) 
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
    return random.sample(captions, num_samples)

prompt_pool = load_coco_prompts(coco_annotation_path, 200)

# -----------------------
# 모델 로드 및 RTX 6000 최적화
# -----------------------
print("[*] Loading model for RTX 6000...")
# RTX 6000(Ada)은 bfloat16에서 최상의 성능을 냅니다.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.bfloat16, 
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 1. Attention 최적화
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers Enabled")
except:
    print("[*] Using PyTorch SDPA (Default)")

# 2. 핵심: torch.compile 적용 (Throughput 병목 해결의 핵심)
# RTX 6000급 성능에서는 'reduce-overhead' 모드가 가장 효과적입니다.
print("[*] Compiling UNet (Optimizing for RTX 6000)...")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Warm-up (컴파일 시간 포함)
# -----------------------
print("[*] Warm-up and Compiling... (Wait about 2-3 min)")
with torch.inference_mode():
    for _ in range(3): 
        _ = pipe(prompt_pool[:2], num_inference_steps=20)
torch.cuda.synchronize()
print("[*] Compilation Complete.")

# -----------------------
# CSV 결과 파일 초기화
# -----------------------
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Run", "Batch", "Steps", "Batch_Latency_s", "User_Latency_s", 
        "Throughput_img_s", "Peak_Mem_GB", "GPU_Util_avg_%"
    ])

print("\nRun | Batch | Step | BatchLat | UserLat | Throughput | PeakMem | GPU Util")
print("-" * 100)

# -----------------------
# 메인 실험 루프
# -----------------------
for run in range(1, num_runs + 1):
    for T in step_sizes:
        for B in batch_sizes:
            try:
                # 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                prompts = prompt_pool[:B]
                
                # 측정 시작
                monitor.start()
                start = time.time()

                with torch.inference_mode():
                    # TF32 등 하드웨어 가속 활성화 (RTX 6000 전용)
                    torch.set_float32_matmul_precision('high')
                    output = pipe(prompts, num_inference_steps=T)

                torch.cuda.synchronize()
                end = time.time()
                avg_gpu_util = monitor.stop()

                # 지표 계산
                batch_latency = end - start
                user_latency = batch_latency / B
                throughput = B / batch_latency
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                print(f"{run:<3} | {B:<5} | {T:<4} | {batch_latency:<8.2f} | {user_latency:<7.3f} | {throughput:<10.2f} | {peak_mem:<7.2f} | {avg_gpu_util:<6.1f}")

                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, batch_latency, user_latency, throughput, peak_mem, avg_gpu_util])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{run:<3} | {B:<5} | {T:<4} | OOM (Out of Memory)")
                    monitor.stop()
                    break 
                else:
                    monitor.stop()
                    raise e

print(f"\n[✔] 실험 완료! 결과: {os.path.abspath(csv_output_file)}")
