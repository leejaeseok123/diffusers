import sys
import torch
import json
import random
import csv
import os
import gc
import time
import numpy as np
from diffusers import FluxPipeline, StableDiffusion3Pipeline

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================

H, W = 1024, 1024
BATCH_SIZE = 1  # 배치 사이즈 1 고정
NUM_WARMUP_RUNS = 3  # 웜업 횟수
NUM_BENCH_RUNS = 5   # latency 측정 반복 횟수 (프롬프트 5개 사용해 평균 산출)
SEED = 42

GUIDANCE_SCALE_FLUX = 3.5
GUIDANCE_SCALE_TURBO = 0.0  # SD3.5-Turbo 기본값 CFG=0.0

# 지정된 step 리스트
step_sizes = [4, 6, 8, 12, 14, 16, 18, 20, 30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"

coco_annotation_path = (
    "/home/jslee/diffusion_exper/batch_exper/dataset/"
    "coco2014/annotation/captions_val2014.json"
)

# 실험 대상 모델 정보 정의
MODELS = [
    {
        "version": "flux1-dev",
        "id": "/mnt/ssd1/jslee/huggingface/hub/FLUX.1-dev-full",
        "type": "flux",
        "dtype": torch.bfloat16,
        "guidance": GUIDANCE_SCALE_FLUX,
    },
    {
        "version": "sd3.5-turbo",
        "id": "stabilityai/stable-diffusion-3.5-large-turbo",
        "type": "sd3_turbo",
        "dtype": torch.float16,
        "guidance": GUIDANCE_SCALE_TURBO,
    },
]

# ============================================================
# Utils
# ============================================================

def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, "r") as f:
        data = json.load(f)
    captions = sorted(
        list(set([ann["caption"] for ann in data["annotations"]]))
    )
    return captions[:n]

# Latency 측정용으로 사용할 COCO 프롬프트 로드
prompt_pool = load_coco_prompts(coco_annotation_path, NUM_BENCH_RUNS)

# ============================================================
# Benchmark Loop for Each Model
# ============================================================

for m_cfg in MODELS:
    version = m_cfg["version"]
    model_id = m_cfg["id"]
    model_type = m_cfg["type"]
    dtype = m_cfg["dtype"]
    guidance = m_cfg["guidance"]

    csv_output_file = f"{base_path}/results/{version}_latency_b1.csv"
    os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

    print(f"\n============================================================")
    print(f"[INFO] Starting Latency Benchmark for Model: {version}")
    print(f"============================================================")

    # 1. 모델 로드
    if model_type == "flux":
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to("cuda")
    elif model_type == "sd3_turbo":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=dtype
        ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)

    # 2. Warm-up
    print(f"[INFO] Warm-up for {version} ({NUM_WARMUP_RUNS} runs)...")
    with torch.inference_mode():
        for _ in range(NUM_WARMUP_RUNS):
            _ = pipe(
                prompt=prompt_pool[:1],
                num_inference_steps=20,
                guidance_scale=guidance,
                height=H,
                width=W,
            )
    torch.cuda.synchronize()
    print("[INFO] Warm-up done!\n")

    # 3. CSV 초기화
    with open(csv_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Steps",
            "Mean_Latency_sec",
            "Std_Latency_sec",
            "Resolution",
            "BatchSize",
            "Runs",
        ])

    print(f"{'Steps':<8} | {'Mean Latency (s)':<18} | {'Std (s)':<10}")
    print("-" * 45)

    # 4. Step별 Latency 측정 Loop
    for T in step_sizes:
        try:
            torch.cuda.empty_cache()
            gc.collect()

            latencies = []

            # COCO 프롬프트 5개를 순회하며 5회 반복 측정
            for run_idx in range(NUM_BENCH_RUNS):
                current_prompt = [prompt_pool[run_idx]]
                generator = torch.Generator(device="cuda").manual_seed(SEED + run_idx)

                torch.cuda.synchronize()
                start_time = time.time()

                with torch.inference_mode():
                    _ = pipe(
                        prompt=current_prompt,
                        num_inference_steps=T,
                        guidance_scale=guidance,
                        height=H,
                        width=W,
                        generator=generator,
                    )

                torch.cuda.synchronize()
                end_time = time.time()

                latencies.append(end_time - start_time)

            mean_latency = float(np.mean(latencies))
            std_latency = float(np.std(latencies))

            print(f"{T:<8} | {mean_latency:<18.4f} | {std_latency:<10.4f}")

            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    T,
                    f"{mean_latency:.4f}",
                    f"{std_latency:.4f}",
                    f"{H}x{W}",
                    BATCH_SIZE,
                    NUM_BENCH_RUNS,
                ])

        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"{T:<8} | OOM")
                with open(csv_output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([T, "OOM", "OOM", f"{H}x{W}", BATCH_SIZE, NUM_BENCH_RUNS])
            else:
                print(f"{T:<8} | ERROR: {e}")
                with open(csv_output_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([T, "ERROR", "ERROR", f"{H}x{W}", BATCH_SIZE, NUM_BENCH_RUNS])

            torch.cuda.empty_cache()
            gc.collect()

    # 다음 모델 로드를 위한 메모리 해제
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[INFO] Completed benchmark for {version}. Pipeline cleared.")

print("\n[SUCCESS] All latency benchmarks finished successfully!")
