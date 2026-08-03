"""
PaniniServe — 실제 GPU 실행 스크립트
======================================
주신 SD v2.1 스케일링 벤치마크 스크립트의 구조(재현성 고정, GPU 모니터링 스레드,
warm-up, CSV 출력)를 그대로 따라서, PaniniServe 스케줄러가 결정한 스케줄대로
실제 SDv2.1 / SDv3.5-Medium을 호출해 COCO 프롬프트로 이미지를 생성합니다.

흐름: (1) 요청 300개 생성(포아송 도착 + accuracy/model 랜덤 배정)
      (2) PaniniServe 스케줄러가 "시뮬레이션"으로 순서/모델/step을 미리 결정
      (3) 그 결정대로 실제 GPU에서 하나씩 실행 (원래 도착시각에 맞춰 sleep,
          real-time replay로 실제 큐잉/부하를 재현)
      (4) 요청마다 CSV 한 줄 기록 (latency, SLO 만족 여부, GPU 상태 등)
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
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline, DDIMScheduler
from pynvml import *

sys.stdout.reconfigure(line_buffering=True)

# =====================================================================
# 재현성 고정 (참고 스크립트와 동일)
# =====================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================================================================
# GPU 모니터링 (참고 스크립트와 동일한 클래스)
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
csv_output_file = "PaniniServe_medium.csv"

TOTAL_REQUESTS = 300
LOAD_LEVEL = "medium"          # "low" | "medium" | "high" -- 돌릴 때마다 바꿔서 실행
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

# --- Model Profile (사전 프로파일링 값 -- 스케줄링 "결정"에만 쓰이는 예측치.
#     실제 측정치는 실행 후 CSV의 System_Latency_s 컬럼에 별도로 기록됨) ---
TIME_PER_STEP = {"small": 0.08093, "large": 0.39624}
BASE_OVERHEAD = {"small": 0.09463, "large": 0.27221}
SWITCH_TIME = {("small", "large"): 1.90, ("large", "small"): 6.31}


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
    start: Optional[float] = None          # 예측(시뮬레이션) 시작시각 -> 실행 후 실측치로 덮어씀
    end: Optional[float] = None            # 예측 종료시각 -> 실측치로 덮어씀
    dropped: bool = False
    switched: bool = False
    status: Optional[str] = None
    overage_pct: Optional[float] = None

    @property
    def accuracy_met(self) -> bool:
        return (not self.dropped) and self.assigned_steps is not None \
            and self.assigned_steps >= self.min_steps


# =====================================================================
# 데이터 로드 -- 참고 스크립트와 동일한 방식(중복 제거+정렬)으로 COCO 프롬프트 300개 확보
# =====================================================================

def load_coco_prompts(json_path: str, num_samples: int) -> List[str]:
    print("[*] Loading COCO prompts...")
    with open(json_path, "r") as f:
        data = json.load(f)
    captions = list(set([ann["caption"] for ann in data["annotations"]]))
    captions = sorted(captions)
    return captions[:num_samples]


def generate_requests(load_level: str, prompt_pool: List[str]) -> List[Req]:
    """포아송 도착 + model/accuracy 랜덤 배정으로 요청 300개 생성.
    prompt_pool은 load_coco_prompts()로 미리 로드해둔 걸 그대로 받아서,
    요청 <-> 프롬프트를 중복 없이 1:1로 섞어서 배정한다."""
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
# PaniniServe 스케줄러 (시뮬레이션으로 순서/모델/step을 미리 결정)
#   -- 로직은 panini_serve.py와 동일, 실제 실행 전에 "계획"만 세우는 단계
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
    """PaniniServe 윈도우 스케줄링을 시뮬레이션으로 돌려서, 각 요청의
    assigned_model/assigned_steps/처리순서(=이 함수가 반환하는 리스트 순서)와
    drop 여부까지 전부 미리 "계획"만 세운다. 아직 실제 GPU 호출은 하지 않는다."""
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

        # (예측 start/end는 여기서 채워두지만, 실제 실행 후 real_start/real_end로 덮어씀)
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

    # rid 순서가 아니라 "실제로 처리되는 순서"로 정렬된 상태로 반환
    # (drop된 요청은 drop된 시점 순서로 섞여 들어가 있으므로, 실행 시점엔
    #  arrival 순서를 기준으로 다시 정렬해서 재현하는 게 자연스러움 -> 아래서 처리)
    planned.sort(key=lambda r: (r.dropped, r.start if r.start is not None else r.arrival))
    return planned


# =====================================================================
# 모델 로드 (참고 스크립트 스타일: DDIM + attention slicing + xformers, 해상도別)
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
    """model_name을 GPU로 올리고, 이전에 GPU에 있던 다른 모델은 CPU로 내린다.
    실제 걸린 스위칭 시간(초)을 반환 -- 이게 SWITCH_TIME 실측치를 얻는 방법이다."""
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
    print("[*] Warm-up 완료! 실험 시작합니다.\n")


# =====================================================================
# 실제 GPU 실행 (real-time replay: 원래 도착시각에 맞춰 sleep 하며 재현)
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
            "Switch_Time_s", "Status", "Overage_pct",
            "Peak_Mem_GB", "GPU_Util_%",
        ])

    print(f"{'RID':<5} | {'Model':<6} | {'Steps':<5} | {'Latency_s':<10} | {'Switch_s':<9} | {'Status':<8} | {'GPU%':<6}")
    print("-" * 70)

    for r in plan:
        if r.dropped:
            # drop된 요청도 CSV엔 남겨서 최종 SLO 계산에 포함시킴 (latency/gpu 정보는 없음)
            with open(csv_output_file, "a", newline="") as f:
                csv.writer(f).writerow([r.rid, r.assigned_model, r.assigned_steps,
                                          r.arrival, r.deadline, "", "", "", "",
                                          "dropped", "", "", ""])
            continue

        if realtime:
            wait = r.arrival - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

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
        r.status = "met" if r.end <= r.deadline else "soft"
        if r.status == "soft":
            r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            image.save(os.path.join(save_dir, f"req_{r.rid:03d}.png"))

        system_latency = real_end - real_start
        print(f"{r.rid:<5} | {r.assigned_model:<6} | {r.assigned_steps:<5} | "
              f"{system_latency:<10.4f} | {switch_time:<9.4f} | {r.status:<8} | {gpu_util:<6.1f}")

        with open(csv_output_file, "a", newline="") as f:
            csv.writer(f).writerow([
                r.rid, r.assigned_model, r.assigned_steps, r.arrival, r.deadline,
                real_start, real_end, system_latency, switch_time,
                r.status, r.overage_pct or "", peak_mem, gpu_util,
            ])

    print(f"\n[✔] 완료 -> {os.path.abspath(csv_output_file)}")


# =====================================================================
# 실행부
# =====================================================================

if __name__ == "__main__":
    prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_REQUESTS)
    reqs = generate_requests(LOAD_LEVEL, prompt_pool)
    plan = plan_schedule(reqs)   # (1) 시뮬레이션으로 스케줄 먼저 결정

    pipes = load_pipelines()
    warm_up(pipes, prompt_pool)

    # 먼저 소수(예: 10개)만 대기 없이 빠르게 돌려서 프로파일 값 검증하고 싶으면:
    # run_real_execution(plan[:10], pipes, realtime=False)

    # 논문용 실제 결과 (원래 도착시각에 맞춰 sleep, 실제 부하 재현):
    run_real_execution(plan, pipes, realtime=True, save_dir=f"outputs/panini_{LOAD_LEVEL}")
