"""
PaniniServe — 실제 GPU 실행 스크립트 (SLO 및 Throughput 종합 리포팅 적용)
"""

import sys
import os
import json
import time
import random
import csv
import gc
import threading
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# =====================================================================
# 재현성 고정
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================================================================
# GPU 모니터링
# =====================================================================
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
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.thread.join()
        if not self.utils:
            return 0
        return sum(self.utils) / len(self.utils)


monitor = GPUUtilMonitor(handle)

# =====================================================================
# 실험 설정
# =====================================================================
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "PaniniServe_low.csv"

TOTAL_REQUESTS = 300
LOAD_LEVEL = "low"          # "low" | "medium" | "high" -- 실행 시 조절
WINDOW_SEC = 60.0

RES = {"small": (768, 768), "large": (1024, 1024)}   # SDv2.1 / SDv3.5-Medium 해상도

MODEL_IDS = {
    "small": "Manojb/stable-diffusion-2-1-base",
    "large": "stabilityai/stable-diffusion-3.5-medium",
}

ACCURACY_STEP_RANGE = {"high": (30, 40), "medium": (20, 30), "low": (10, 20)}
LOAD_LEVELS = {"low": (1.0, 2.0), "medium": (3.0, 4.0), "high": (10.0, 18.0)}
SLO_MULTIPLIER_RANGE = (2.0, 3.0)
STEP_RANGE_BOUNDS = (10, 40)
MODEL_TYPES = ["small", "large", "auto"]
ACCURACY_LEVELS = ["high", "medium", "low"]
INITIAL_MODEL = "small"

# --- Accuracy SLO용 부하 조건별 최소 요구 Step 기준 ---
MIN_STEPS_THRESHOLD = {
    "low": 10,
    "medium": 20,
    "high": 30
}

# --- Model Profile ---
TIME_PER_STEP = {"small": 0.08093, "large": 0.39624}
BASE_OVERHEAD = {"small": 0.09463, "large": 0.27221}
SWITCH_TIME = {("small", "large"): 2.30, ("large", "small"): 4.50}


def exec_time(model: str, steps: float) -> float:
    return BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]


def switch_cost(prev_model, next_model) -> float:
    if prev_model is None or prev_model == next_model:
        return 0.0
    return SWITCH_TIME[(prev_model, next_model)]


def slo_multiplier(steps: float) -> float:
    lo, hi = STEP_RANGE_BOUNDS
    m_lo, m_hi = SLO_MULTIPLIER_RANGE
    frac = (steps - lo) / (hi - lo) if hi > lo else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return m_lo + frac * (m_hi - m_lo)


# =====================================================================
# Request 데이터 모델
# =====================================================================

@dataclass
class Req:
    rid: int
    arrival: float
    prompt: str
    model_pref: str
    accuracy: str
    default_steps: int
    min_steps: int
    calc_model: str
    latency_slo: float
    deadline: float
    assigned_model: Optional[str] = None
    assigned_steps: Optional[int] = None
    start: Optional[float] = None          # 실측치로 덮어씀
    end: Optional[float] = None            # 실측치로 덮어씀
    dropped: bool = False
    switched: bool = False
    status: Optional[str] = None
    overage_pct: Optional[float] = None

    @property
    def accuracy_met(self) -> bool:
        return (not self.dropped) and self.assigned_steps is not None \
            and self.assigned_steps >= self.min_steps


# =====================================================================
# 데이터 로드
# =====================================================================

def load_coco_prompts(json_path: str, num_samples: int) -> List[str]:
    print("[*] Loading COCO prompts...")
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = list(set([ann["caption"] for ann in data["annotations"]]))
    captions = sorted(captions)
    return captions[:num_samples]


def generate_requests(load_level: str, prompt_pool: List[str]) -> List[Req]:
    prompts = prompt_pool.copy()
    random.shuffle(prompts)

    lo_rate, hi_rate = LOAD_LEVELS[load_level]
    reqs: List[Req] = []
    t = 0.0
    for rid in range(TOTAL_REQUESTS):
        rate_per_min = random.uniform(lo_rate, hi_rate)
        t += random.expovariate(rate_per_min / 60.0)

        model_pref = random.choice(MODEL_TYPES)
        accuracy = random.choice(ACCURACY_LEVELS)
        mn, mx = ACCURACY_STEP_RANGE[accuracy]
        calc_model = "large" if model_pref == "auto" else model_pref
        latency_slo = exec_time(calc_model, mx) * slo_multiplier(mx)
        deadline = t + latency_slo

        reqs.append(Req(
            rid=rid, arrival=round(t, 3), prompt=prompts[rid],
            model_pref=model_pref, accuracy=accuracy,
            default_steps=mx, min_steps=mn, calc_model=calc_model,
            latency_slo=round(latency_slo, 3), deadline=round(deadline, 3),
        ))
    return reqs


# =====================================================================
# PaniniServe 스케줄러 (시뮬레이션)
# =====================================================================

def order_batch(batch: List[Req], current_model: str) -> List[Req]:
    front = [r for r in batch if r.model_pref == current_model]
    other_model = "large" if current_model == "small" else "small"
    back = [r for r in batch if r.model_pref == other_model]
    auto_reqs = [r for r in batch if r.model_pref == "auto"]
    front.sort(key=lambda r: r.deadline)
    back.sort(key=lambda r: r.deadline)
    for r in front:
        r.assigned_model = current_model
    for r in back:
        r.assigned_model = other_model
    queue = front + back
    for req in sorted(auto_reqs, key=lambda r: r.deadline):
        best_pos = None
        for pos in range(len(queue), -1, -1):
            left = queue[pos - 1].assigned_model if pos > 0 else current_model
            right = queue[pos].assigned_model if pos < len(queue) else None
            req.assigned_model = left or right or current_model
            t = 0.0
            prev = current_model
            for r in queue[:pos] + [req]:
                t += switch_cost(prev, r.assigned_model)
                t += exec_time(r.assigned_model, r.default_steps)
                prev = r.assigned_model
            if t <= req.latency_slo:
                best_pos = pos
                break
        if best_pos is None:
            best_pos = len(queue)
            req.assigned_model = queue[-1].assigned_model if queue else current_model
        queue = queue[:best_pos] + [req] + queue[best_pos:]
    return queue


def total_window_time(queue, current_model, use_assigned=False) -> float:
    t, prev = 0.0, current_model
    for r in queue:
        t += switch_cost(prev, r.assigned_model)
        steps = r.assigned_steps if (use_assigned and r.assigned_steps is not None) else r.default_steps
        t += exec_time(r.assigned_model, steps)
        prev = r.assigned_model
    return t


def plan_schedule(reqs: List[Req]) -> List[Req]:
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting: List[Req] = []
    current_model = INITIAL_MODEL
    window_start, current_time = 0.0, 0.0
    planned: List[Req] = []

    while len(planned) < len(reqs):
        window_end = window_start + WINDOW_SEC
        arrived = [r for r in pending if r.arrival <= window_end]
        for r in arrived:
            waiting.append(r)
            pending.remove(r)
        if not waiting:
            window_start = window_end
            current_time = max(current_time, window_start)
            continue

        batch = [r for r in waiting if r.deadline <= window_end]
        if not batch:
            batch = [min(waiting, key=lambda r: r.deadline)]

        queue = order_batch(batch, current_model)
        for r in queue:
            r.assigned_steps = r.default_steps

        start_t = max(current_time, window_start)
        max_allowed = window_end - start_t
        if max_allowed <= 0:
            for r in queue:
                r.dropped, r.status = True, "dropped"
                if r in waiting:
                    waiting.remove(r)
                planned.append(r)
            window_start = window_end
            continue

        total = total_window_time(queue, current_model, use_assigned=True)
        if total > max_allowed:
            overflow = total - max_allowed
            reduce_each = overflow / len(queue)
            for r in queue:
                step_reduction = math.floor(reduce_each / TIME_PER_STEP[r.assigned_model])
                r.assigned_steps = max(r.default_steps - step_reduction, r.min_steps)
            total = total_window_time(queue, current_model, use_assigned=True)

        if total > max_allowed:
            for r in sorted(queue, key=lambda r: -r.deadline):
                if total <= max_allowed:
                    break
                queue.remove(r)
                if r in waiting:
                    waiting.remove(r)
                r.dropped, r.status = True, "dropped"
                planned.append(r)
                total = total_window_time(queue, current_model, use_assigned=True)

        t, prev = start_t, current_model
        for r in queue:
            sc = switch_cost(prev, r.assigned_model)
            r.switched = sc > 0
            t += sc
            r.start = t
            t += exec_time(r.assigned_model, r.assigned_steps)
            r.end = t
            prev = r.assigned_model

        if queue:
            current_model = queue[-1].assigned_model
            current_time = t
        for r in queue:
            if r in waiting:
                waiting.remove(r)
        planned.extend(queue)
        window_start = window_end

    planned.sort(key=lambda r: (r.dropped, r.start if r.start is not None else r.arrival))
    return planned


# =====================================================================
# 모델 로드 및 Warm-up
# =====================================================================

def load_pipelines():
    print("[*] Loading SD v2.1 (768x768)...")
    pipe_small = StableDiffusionPipeline.from_pretrained(
        MODEL_IDS["small"], torch_dtype=torch.float16, safety_checker=None
    )
    pipe_small.scheduler = DDIMScheduler.from_config(pipe_small.scheduler.config)
    pipe_small.enable_attention_slicing()
    try:
        pipe_small.enable_xformers_memory_efficient_attention()
        print("[*] xformers ON (SDv2.1)")
    except Exception:
        print("[!] xformers 없음 (SDv2.1)")
    pipe_small.set_progress_bar_config(disable=True)

    print("[*] Loading SD3.5 Medium (1024x1024)...")
    pipe_large = StableDiffusion3Pipeline.from_pretrained(
        MODEL_IDS["large"], torch_dtype=torch.bfloat16
    )
    pipe_large.set_progress_bar_config(disable=True)

    return {"small": pipe_small, "large": pipe_large}


def switch_to(pipes: dict, model_name: str, current_on_gpu: Optional[str]) -> float:
    if current_on_gpu == model_name:
        return 0.0
    t0 = time.time()
    if current_on_gpu is not None:
        pipes[current_on_gpu].to("cpu", silence_dtype_warnings=True)
    pipes[model_name].to(device)
    torch.cuda.synchronize()
    return time.time() - t0


def warm_up(pipes: dict, prompt_pool: List[str]):
    print("[*] Warm-up 중...")
    switch_to(pipes, "small", None)
    with torch.inference_mode():
        H, W = RES["small"]
        _ = pipes["small"](prompt_pool[:2], num_inference_steps=20, height=H, width=W)
    torch.cuda.synchronize()

    switch_to(pipes, "large", "small")
    with torch.inference_mode():
        H, W = RES["large"]
        _ = pipes["large"](prompt_pool[:2], num_inference_steps=20, height=H, width=W)
    torch.cuda.synchronize()

    print("[*] Warm-up 완료!\n")


# =====================================================================
# 실제 GPU 실행 (Real-time Replay 및 종합 리포팅)
# =====================================================================

def run_real_execution(plan: List[Req], pipes: dict, realtime: bool = True,
                       save_dir: Optional[str] = None):
    current_on_gpu = None
    t0 = time.time()

    with open(csv_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "RID", "Model", "Steps", "Arrival_s", "Deadline_s",
            "Real_Start_s", "Real_End_s", "System_Latency_s",
            "Switch_Time_s", "Status", "Latency_SLO_Met", "Accuracy_SLO_Met",
            "Overage_pct", "Peak_Mem_GB", "GPU_Util_%"
        ])

    print(f"{'RID':<5} | {'Model':<6} | {'Steps':<5} | {'System_Lat_s':<12} | {'Switch_s':<9} | {'Status':<8} | {'GPU%':<6}")
    print("-" * 75)

    for r in plan:
        if r.dropped:
            with open(csv_output_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    r.rid, r.assigned_model, r.assigned_steps,
                    r.arrival, r.deadline, "", "", "", "",
                    "dropped", False, False, "", "", ""
                ])
            continue

        if realtime:
            wait = r.arrival - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

        # 기존 메모리 및 GC 정리 유지
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor.start()
        switch_time = switch_to(pipes, r.assigned_model, current_on_gpu)
        current_on_gpu = r.assigned_model

        real_start = time.time() - t0
        H, W = RES[r.assigned_model]
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        with torch.inference_mode():
            image = pipes[r.assigned_model](
                r.prompt, num_inference_steps=r.assigned_steps,
                height=H, width=W, generator=generator,
            ).images[0]
        torch.cuda.synchronize()
        real_end = time.time() - t0
        gpu_util = monitor.stop()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        r.start, r.end = real_start, real_end
        
        # --- 지표 계산 ---
        system_latency = real_end - r.arrival   # 전체 대기 포함 latency
        latency_slo_met = r.end <= r.deadline    # Latency SLO (met만 True)
        
        min_required_step = MIN_STEPS_THRESHOLD.get(LOAD_LEVEL, 20)
        accuracy_slo_met = r.assigned_steps >= min_required_step  # Accuracy SLO

        r.status = "met" if latency_slo_met else "soft"
        if r.status == "soft":
            r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            image.save(os.path.join(save_dir, f"req_{r.rid:03d}.png"))

        print(f"{r.rid:<5} | {r.assigned_model:<6} | {r.assigned_steps:<5} | "
              f"{system_latency:<12.4f} | {switch_time:<9.4f} | {r.status:<8} | {gpu_util:<6.1f}")

        with open(csv_output_file, "a", newline="") as f:
            csv.writer(f).writerow([
                r.rid, r.assigned_model, r.assigned_steps, r.arrival, r.deadline,
                real_start, real_end, system_latency, switch_time,
                r.status, latency_slo_met, accuracy_slo_met,
                r.overage_pct or "", peak_mem, gpu_util
            ])

    exp_total_time = time.time() - t0
    print(f"\n[✔] 실행 완료 -> {os.path.abspath(csv_output_file)}")

    # =====================================================================
    # 최종 결과 종합 분석, 터미널 출력 및 요약 CSV 저장
    # =====================================================================
    df = pd.read_csv(csv_output_file)
    total_reqs = len(df)
    completed_reqs = len(df[df["Status"] != "dropped"])

    # 1. Throughput
    req_throughput = completed_reqs / exp_total_time

    # 2. Latency SLO
    latency_met_cnt = (df["Latency_SLO_Met"] == True).sum()
    latency_slo_rate = (latency_met_cnt / total_reqs) * 100
    p50_lat = df["System_Latency_s"].dropna().quantile(0.50)
    p95_lat = df["System_Latency_s"].dropna().quantile(0.95)
    p99_lat = df["System_Latency_s"].dropna().quantile(0.99)

    # 3. Accuracy SLO
    acc_met_cnt = (df["Accuracy_SLO_Met"] == True).sum()
    acc_slo_rate = (acc_met_cnt / total_reqs) * 100

    # --- 콘솔 출력 ---
    print("\n" + "=" * 65)
    print(f"       PaniniServe Performance Summary Report ({LOAD_LEVEL.upper()})")
    print("=" * 65)
    print(f" [Throughput]")
    print(f"  - Total Elapsed Time     : {exp_total_time:.2f} s")
    print(f"  - Request Throughput     : {req_throughput:.3f} req/s")
    print("-" * 65)
    print(f" [Latency SLO]")
    print(f"  - Latency SLO Attainment : {latency_slo_rate:.2f}% ({latency_met_cnt}/{total_reqs})")
    print(f"  - System Latency P50     : {p50_lat:.3f} s")
    print(f"  - System Latency P95     : {p95_lat:.3f} s")
    print(f"  - System Latency P99     : {p99_lat:.3f} s")
    print("-" * 65)
    print(f" [Accuracy SLO]")
    print(f"  - Min Required Steps     : {MIN_STEPS_THRESHOLD.get(LOAD_LEVEL, 20)} steps")
    print(f"  - Accuracy SLO Attainment: {acc_slo_rate:.2f}% ({acc_met_cnt}/{total_reqs})")
    print("=" * 65 + "\n")

    # --- 요약 리포트 CSV 저장 (추가된 부분) ---
    summary_csv_file = f"PaniniServe_summary_{LOAD_LEVEL}.csv"
    summary_data = {
        "Metric": [
            "Load_Level", "Total_Requests", "Completed_Requests", "Total_Elapsed_Time_s",
            "Throughput_req_s", "Latency_SLO_Attainment_%", "System_Latency_P50_s",
            "System_Latency_P95_s", "System_Latency_P99_s", "Accuracy_SLO_Attainment_%"
        ],
        "Value": [
            LOAD_LEVEL, total_reqs, completed_reqs, round(exp_total_time, 2),
            round(req_throughput, 3), round(latency_slo_rate, 2), round(p50_lat, 3),
            round(p95_lat, 3), round(p99_lat, 3), round(acc_slo_rate, 2)
        ]
    }
    pd.DataFrame(summary_data).to_csv(summary_csv_file, index=False)
    print(f"[✔] 요약 분석 리포트 저장 완료 -> {os.path.abspath(summary_csv_file)}")

# =====================================================================
# 실행부
# =====================================================================

if __name__ == "__main__":
    prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_REQUESTS)
    reqs = generate_requests(LOAD_LEVEL, prompt_pool)
    plan = plan_schedule(reqs)

    pipes = load_pipelines()
    warm_up(pipes, prompt_pool)

    # 논문용 실제 결과 (도착시각에 맞춰 sleep, 실제 부하 재현)
    run_real_execution(plan, pipes, realtime=True, save_dir=f"outputs/panini_{LOAD_LEVEL}")
