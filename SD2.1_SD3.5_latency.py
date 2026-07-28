import sys
import torch
import time
import json
import csv
import os
import gc
import threading
import numpy as np

from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 재현성 고정
# -----------------------
SEED = 42
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
        self.thread.join()
        if not self.utils: return 0
        return sum(self.utils) / len(self.utils)

monitor = GPUUtilMonitor(handle)

# -----------------------
# 공통 실험 설정
# -----------------------
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
base_path = "/home/jslee/diffusion_exper/batch_exper/fid"

BATCH_SIZE = 1
NUM_BENCH_RUNS = 5  # 스텝당 평균 계산을 위한 반복 횟수
step_sizes = list(range(10, 41))  # 10부터 40까지 1스텝 단위

# -----------------------
# COCO 프롬프트 1개 로드
# -----------------------
def load_single_coco_prompt(json_path):
    print("[*] Loading COCO prompt...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[0]

single_prompt = load_single_coco_prompt(coco_annotation_path)

# -----------------------
# 모델 설정 정의
# -----------------------
MODELS = [
    {
        "version": "sd_v2.1",
        "id": "Manojb/stable-diffusion-2-1-base",
        "type": "sd21",
        "dtype": torch.float16,
        "height": 768,
        "width": 768,
    },
    {
        "version": "sd3.5_medium",
        "id": "stabilityai/stable-diffusion-3.5-medium",
        "type": "sd35",
        "dtype": torch.bfloat16,
        "height": 1024,
        "width": 1024,
    },
]

# -----------------------
# 메인 벤치마크 루프
# -----------------------
for m_cfg in MODELS:
    version = m_cfg["version"]
    model_id = m_cfg["id"]
    model_type = m_cfg["type"]
    dtype = m_cfg["dtype"]
    H, W = m_cfg["height"], m_cfg["width"]

    csv_output_file = f"{base_path}/results/{version}_latency_b1_steps10_40.csv"
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    print(f"\n============================================================")
    print(f"[*] Starting Latency Benchmark for: {version} ({H}x{W})")
    print(f"============================================================")

    # 1. 모델 로드
    if model_type == "sd21":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[*] xformers ON")
        except:
            print("[!] xformers 사용 불가 (기본 attention 적용)")

    elif model_type == "sd35":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        ).to(device)
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=True)

    # 2. Warm-up
    print(f"[*] Warm-up 중...")
    with torch.inference_mode():
        _ = pipe([single_prompt], num_inference_steps=20, height=H, width=W)
    torch.cuda.synchronize()
    print("[*] Warm-up 완료!\n")

    # 3. CSV 초기화
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Steps",
            "Mean_Latency_s",
            "Std_Latency_s",
            "Peak_Mem_GB",
            "GPU_Util_%",
            "Resolution",
            "BatchSize",
            "Runs"
        ])

    print(f"{'Steps':<6} | {'Mean Latency (s)':<18} | {'Std (s)':<10} | {'PeakMem (GB)':<12} | {'GPU %':<6}")
    print("-" * 65)

    # 4. Step별 Latency 측정 Loop
    for T in step_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            latencies = []
            
            monitor.start()

            for run_idx in range(NUM_BENCH_RUNS):
                generator = torch.Generator(device=device).manual_seed(SEED + run_idx)

                torch.cuda.synchronize()
                start_time = time.time()

                with torch.inference_mode():
                    _ = pipe(
                        [single_prompt],
                        num_inference_steps=T,
                        height=H,
                        width=W,
                        generator=generator
                    )

                torch.cuda.synchronize()
                latencies.append(time.time() - start_time)

            gpu_util = monitor.stop()
            mean_latency = float(np.mean(latencies))
            std_latency = float(np.std(latencies))
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            print(f"{T:<6} | {mean_latency:<18.4f} | {std_latency:<10.4f} | {peak_mem:<12.2f} | {gpu_util:<6.1f}")

            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    T,
                    f"{mean_latency:.4f}",
                    f"{std_latency:.4f}",
                    f"{peak_mem:.2f}",
                    f"{gpu_util:.1f}",
                    f"{H}x{W}",
                    BATCH_SIZE,
                    NUM_BENCH_RUNS
                ])

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{T:<6} | OOM")
                monitor.stop()
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([T, "OOM", "OOM", "OOM", "OOM", f"{H}x{W}", BATCH_SIZE, NUM_BENCH_RUNS])
            else:
                monitor.stop()
                raise e

            torch.cuda.empty_cache()
            gc.collect()

    # 메모리 해제 후 다음 모델 진행
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[*] {version} 측정 완료 및 메모리 해제.")

nvmlShutdown()
print("\n[✔] 모든 Latency 측정 완료!")
