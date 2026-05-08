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

# 1. 재현성 및 필수 변수 설정
SEED = 42  # NameError 방지를 위해 SEED 정의 추가
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

sys.stdout.reconfigure(line_buffering=True)

# 2. GPU 모니터링 클래스
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
        if hasattr(self, 'thread'):
            self.thread.join()
        if not self.utils: return 0
        return sum(self.utils) / len(self.utils)

monitor = GPUUtilMonitor(handle)

# 3. 실험 환경 설정
device = "cuda"
# 경로가 실제 환경과 맞는지 확인해주세요
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "SD_v1.5_scaling.csv"

total_images = 300
batch_sizes  = [1, 2, 4, 8, 16, 32, 64, 80, 96, 128]
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
H, W         = 512, 512

# 4. 데이터 로드 함수
def load_coco_prompts(json_path, num_samples):
    print("[*] Loading COCO prompts...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:num_samples]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# 5. 모델 로드 (SD v1.5)
print("[*] Loading SD v1.5 (512x512)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 메모리 효율화를 위해 xformers가 설치되어 있다면 활성화 권장
#try:
#    pipe.enable_xformers_memory_efficient_attention()
#    print("[*] xformers ON")
#except Exception:
#    pipe.enable_attention_slicing()
#    print("[*] xformers 없음 - Attention Slicing 사용")

pipe.set_progress_bar_config(disable=True)

# Warm-up
print("[*] Warm-up 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=10, height=H, width=W)
torch.cuda.synchronize()
print("[*] Warm-up 완료! 실험을 시작합니다.\n")

# 6. CSV 초기화
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

print(f"{'Run':<4} | {'Batch':<6} | {'Steps':<6} | {'SystemLat':<11} | {'UserLat':<9} | {'Throughput':<12} | {'PeakMem':<9} | {'GPU%':<6}")
print("-" * 95)

# 7. 메인 실험 루프
run = 1  # num_runs가 1이므로 루프 단순화
for T in step_sizes:
    for B in batch_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            monitor.start()
            start_total = time.time()
            user_latency = 0

            with torch.inference_mode():
                for i in range(0, total_images, B):
                    batch_prompts = prompt_pool[i:i+B]
                    if not batch_prompts: break
                    
                    generator = torch.Generator(device=device).manual_seed(SEED)
                    _ = pipe(batch_prompts, num_inference_steps=T,
                             height=H, width=W, generator=generator).images
                    
                    # 첫 번째 배치가 끝나는 시간을 측정하여 User Latency로 사용
                    if i == 0:
                        torch.cuda.synchronize()
                        user_latency = time.time() - start_total

            torch.cuda.synchronize()
            end_total = time.time()
            gpu_util = monitor.stop()

            # 지표 계산
            total_time = end_total - start_total
            system_latency = total_time / total_images  # 이미지 장당 평균 처리 시간
            throughput = B / user_latency               # 초당 이미지 생성 수 (1개 배치 기준)
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3

            print(f"{run:<4} | {B:<6} | {T:<6} | {system_latency:<11.4f} | {user_latency:<9.4f} | {throughput:<12.2f} | {peak_mem:<9.2f} | {gpu_util:<6.1f}")

            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run, B, T, system_latency, user_latency, throughput, peak_mem, gpu_util])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{run:<4} | {B:<6} | {T:<6} | OOM 발생")
                monitor.stop()
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM", "OOM"])
                torch.cuda.empty_cache()
                gc.collect()
                break # 해당 Steps에서 더 큰 배치는 시도하지 않고 다음 Steps로 이동
            else:
                monitor.stop()
                raise e

print(f"\n[✔] 실험 완료! 결과 저장 위치: {os.path.abspath(csv_output_file)}")
