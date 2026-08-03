"""
PaniniServe: Diffusion Model Serving Simulator
================================================
담당 범위: Request Generator / Execution Time & Model Profile /
           PaniniServe 스케줄러 / Metric 계산 / Figure 산출

Clipper-Large, Clipper-Small, INFaaS (전부 FIFO 기반 baseline)는 팀원 담당이며
이 파일에는 포함하지 않습니다. 대신 팀원 스케줄러가 아래와 동일한 인터페이스
(List[Req] -> List[Req])를 따르면, run_and_report() 아래쪽에 그대로
끼워 넣어서 같은 Metric/Figure 파이프라인을 공유할 수 있습니다.

    def run_clipper_large(reqs: List[Req]) -> List[Req]: ...
    def run_clipper_small(reqs: List[Req]) -> List[Req]: ...
    def run_infaas(reqs: List[Req]) -> List[Req]: ...

모델: SDv2.1(Small), SDv3.5-Medium(Large)
데이터셋: COCO2014 caption 파일에서 프롬프트 300개 추출 (파일 없으면 placeholder)
"""

import json
import math
import os
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Dict
import matplotlib.pyplot as plt


# =====================================================================
# 1. Model Profile — SDv2.1(small) / SDv3.5-Medium(large) 실측 프로파일링 값
# =====================================================================
# 실제 사용 모델 (HuggingFace):
#   small (SDv2.1)      : StableDiffusionPipeline.from_pretrained(
#                            "Manojb/stable-diffusion-2-1-base",
#                            torch_dtype=torch.float16, safety_checker=None)
#   large (SDv3.5-Medium): StableDiffusion3Pipeline.from_pretrained(
#                            "stabilityai/stable-diffusion-3.5-medium",
#                            torch_dtype=torch.bfloat16)
#
# latency(steps) = BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]
# (텍스트 인코딩·VAE 디코딩처럼 step 수와 무관한 고정비용 + step당 비용을
#  실측 데이터에 선형회귀(R^2≈0.9999)로 분리한 값)

TIME_PER_STEP = {
    "small": 0.05535,   # SDv2.1, 초/step
    "large": 0.28980,   # SDv3.5-Medium, 초/step
}

BASE_OVERHEAD = {
    "small": 0.0903,    # SDv2.1, 초 (텍스트 인코딩+VAE 디코딩 등 고정비용)
    "large": 0.2906,    # SDv3.5-Medium, 초
}

# 모델 스위칭 오버헤드 (실측치, 방향에 따라 비대칭)
SWITCH_TIME = {
    ("small", "large"): 1.90,   # SDv2.1 -> SDv3.5-Medium
    ("large", "small"): 6.31,   # SDv3.5-Medium -> SDv2.1
}

INITIAL_MODEL = "small"   # 시뮬레이션 시작 시 GPU에 이미 로딩되어 있다고 가정하는 모델


# =====================================================================
# 2. Execution Time 계산 함수
# =====================================================================

def exec_time(model: str, steps: float) -> float:
    """모델과 step 수를 받아 실제 추론 실행시간(초)을 반환."""
    return BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]


def switch_cost(prev_model: Optional[str], next_model: str) -> float:
    """직전 로딩된 모델(prev_model)에서 next_model로 바꿀 때의 스위칭 오버헤드.
    같은 모델이면 0, 최초 실행(prev_model=None)이면 0."""
    if prev_model is None or prev_model == next_model:
        return 0.0
    return SWITCH_TIME[(prev_model, next_model)]


# =====================================================================
# 3. 실험 파라미터
# =====================================================================

TOTAL_REQUESTS = 300          # 프롬프트 1개 = 요청 1건, 300개 소진 시 종료
WINDOW_SEC = 60.0              # 스케줄링 윈도우 주기 (초)

# accuracy 레벨별 denoising step 구간 (min, max)
ACCURACY_STEP_RANGE = {
    "high":   (30, 40),
    "medium": (20, 30),
    "low":    (10, 20),
}

# latency SLO 배수: 프로파일링 기반, step 10~40 구간에 걸쳐 2.0배~3.0배로 선형 보간
SLO_MULTIPLIER_RANGE = (2.0, 3.0)
STEP_RANGE_BOUNDS = (10, 40)

# 트래픽 부하 3단계 (요청/분, 포아송 프로세스의 순간 도착률)
LOAD_LEVELS = {
    "low":    (1.0, 2.0),
    "medium": (3.0, 4.0),
    "high":   (10.0, 18.0),
}

MODEL_TYPES = ["small", "large", "auto"]     # small=SDv2.1, large=SDv3.5-Medium
ACCURACY_LEVELS = ["high", "medium", "low"]

# COCO2014 캡션 파일 경로 (표준 형식: annotations["annotations"][i]["caption"])
# 파일이 없으면 자동으로 placeholder 프롬프트로 대체됨
COCO_CAPTION_PATH = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"


def slo_multiplier(steps: float) -> float:
    """step 수(10~40)에 따라 latency SLO 배수를 2.0~3.0배로 선형 보간."""
    lo, hi = STEP_RANGE_BOUNDS
    m_lo, m_hi = SLO_MULTIPLIER_RANGE
    frac = (steps - lo) / (hi - lo) if hi > lo else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return m_lo + frac * (m_hi - m_lo)


def load_coco_prompts(n: int, path: str = COCO_CAPTION_PATH) -> List[str]:
    """COCO2014 caption 파일에서 프롬프트 n개를 무작위로 뽑아 반환.
    파일이 없으면 placeholder 프롬프트(prompt_000 ~)로 대체."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        captions = [ann["caption"].strip() for ann in data.get("annotations", [])]
        if len(captions) >= n:
            return random.sample(captions, n)
        print(f"[warn] COCO 캡션이 {len(captions)}개뿐이라 {n}개를 못 채움 -> placeholder로 채움")
        captions += [f"placeholder_prompt_{i:03d}" for i in range(n - len(captions))]
        return captions[:n]
    else:
        print(f"[warn] COCO 캡션 파일을 못 찾음 ({path}) -> placeholder 프롬프트 사용")
        return [f"placeholder_prompt_{i:03d}" for i in range(n)]


# =====================================================================
# 4. Request 데이터 모델
# =====================================================================

@dataclass
class Req:
    rid: int
    arrival: float             # 도착 시각 (초)
    prompt: str                 # COCO 캡션 (또는 placeholder)
    model_pref: str              # "small" / "large" / "auto"
    accuracy: str                 # "high" / "medium" / "low"
    default_steps: int            # accuracy 레벨의 최대 step (요청 시 디폴트값)
    min_steps: int                 # accuracy 레벨의 최소 step (스텝 축소 시 하한)
    calc_model: str                 # latency SLO 계산에 쓴 모델 (auto는 'large' 기준, 보수적 가정)
    latency_slo: float                # 허용 latency (초)
    deadline: float                    # 처리 마감 시각 = arrival + latency_slo

    # 아래는 스케줄링/실행 이후 채워지는 필드
    assigned_model: Optional[str] = None
    assigned_steps: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    dropped: bool = False
    switched: bool = False              # 이 요청 처리 직전에 모델 스위칭이 발생했는지
    status: Optional[str] = None         # "met" / "soft" / "dropped"
    overage_pct: Optional[float] = None   # soft일 때, deadline을 몇 % 초과했는지

    @property
    def accuracy_met(self) -> bool:
        """accuracy SLO 만족 여부: drop 안 됐고, 실제 실행 step이 그 레벨 최소값 이상."""
        return (not self.dropped) and self.assigned_steps is not None \
            and self.assigned_steps >= self.min_steps


# =====================================================================
# 5. Request Generator — 포아송 도착 + COCO 프롬프트 + model/accuracy 랜덤 배정
# =====================================================================

def generate_requests(load_level: str, seed: Optional[int] = None) -> List[Req]:
    """지정한 부하 수준(load_level)으로 포아송 프로세스에 따라 요청 300개를 생성.
    프롬프트는 COCO2014 캡션(또는 placeholder)에서 중복 없이 300개를 뽑아 1:1 배정.
    latency SLO는 사용자가 정하는 게 아니라, 그 요청의 accuracy 레벨 기준 최대 step
    실행시간의 2.0~3.0배로 시스템이 자동 계산."""
    if seed is not None:
        random.seed(seed)

    lo_rate, hi_rate = LOAD_LEVELS[load_level]   # 분당 요청 수 범위
    prompts = load_coco_prompts(TOTAL_REQUESTS)
    random.shuffle(prompts)   # 순서 섞어서 프롬프트 <-> 요청을 1:1로 중복 없이 배정

    reqs: List[Req] = []
    t = 0.0
    for rid in range(TOTAL_REQUESTS):
        # 이번 요청의 도착 간격을 지수분포에서 샘플링 (포아송 프로세스의 성질)
        rate_per_min = random.uniform(lo_rate, hi_rate)
        rate_per_sec = rate_per_min / 60.0
        t += random.expovariate(rate_per_sec)

        model_pref = random.choice(MODEL_TYPES)
        accuracy = random.choice(ACCURACY_LEVELS)
        min_steps, max_steps = ACCURACY_STEP_RANGE[accuracy]

        # auto 요청의 latency SLO는 보수적으로 large(느린 쪽) 모델 기준으로 계산
        calc_model = "large" if model_pref == "auto" else model_pref
        latency_slo = exec_time(calc_model, max_steps) * slo_multiplier(max_steps)
        deadline = t + latency_slo

        reqs.append(Req(
            rid=rid,
            arrival=round(t, 3),
            prompt=prompts[rid],
            model_pref=model_pref,
            accuracy=accuracy,
            default_steps=max_steps,
            min_steps=min_steps,
            calc_model=calc_model,
            latency_slo=round(latency_slo, 3),
            deadline=round(deadline, 3),
        ))
    return reqs


def clone_requests(reqs: List[Req]) -> List[Req]:
    """같은 요청 세트를 여러 스케줄러(PaniniServe / Clipper / INFaaS)에 공평하게
    적용하기 위해, 스케줄링 결과로 오염되지 않은 깨끗한 복사본을 만든다."""
    return [Req(rid=r.rid, arrival=r.arrival, prompt=r.prompt, model_pref=r.model_pref,
                 accuracy=r.accuracy, default_steps=r.default_steps, min_steps=r.min_steps,
                 calc_model=r.calc_model, latency_slo=r.latency_slo, deadline=r.deadline)
            for r in reqs]


# =====================================================================
# 6. 스케줄러 — policy 단위로 분리 (PaniniServe / Clipper / INFaaS 공용 골격)
# =====================================================================
# 실행부(exec_time, switch_cost, met/soft 판정, throughput 계산)는 정책과
# 무관하게 항상 동일합니다. 정책별로 다른 부분만 아래 4개 함수로 분리했습니다:
#
#   기능        | PaniniServe      | Clipper           | INFaaS
#   -----------|-------------------|--------------------|-------------------
#   Queue 정렬  | EDF + locality    | FIFO               | FIFO
#   모델 선택   | 동적(스위칭 최소화)| 고정(large 또는 small)| 동적(요청 선호 존중)
#   step        | 가변              | 고정               | 고정
#   step 감소   | O                 | X                  | X
#   drop        | O                 | O (동일 메커니즘)   | O (동일 메커니즘)
#
# 팀원 Clipper/INFaaS 쪽 담당자는 여기 SCHEDULING_POLICIES에 자기 정책 이름을
# 등록하고 assign_model()/assign_steps()에 분기만 추가하면, run_scheduler()의
# 윈도우/drop 로직을 그대로 재사용할 수 있습니다.

BASELINE_STEP_BY_ACCURACY = {"low": 15, "medium": 25, "high": 35}   # 각 accuracy 레벨의 평균(중간) step
# Clipper/INFaaS는 step 조절 기능이 없지만, 그렇다고 모든 요청에 똑같은 값을
# 쓰면 accuracy SLO가 사실상 항상 100%로 나와 변별력이 없어지는 문제가 있었음.
# 그래서 "그때그때 요청의 accuracy 레벨에 맞는 평균 step"으로 배정하기로 확정.


def fifo(batch: List[Req]) -> List[Req]:
    """Clipper/INFaaS용 단순 FIFO 정렬. 도착한 순서 그대로만 정렬하고,
    deadline이나 모델 종류를 고려한 재정렬은 전혀 하지 않는다."""
    return sorted(batch, key=lambda r: r.arrival)


def order_batch(batch: List[Req], current_model: str) -> List[Req]:
    """PaniniServe 전용: EDF(Earliest Deadline First) + locality-aware 정렬.
    1) 현재 GPU에 로딩된 모델(current_model)과 같은 모델을 원하는 요청
       -> deadline 빠른 순으로 앞쪽 블록(front)에 배치
    2) 다른 모델을 원하는 요청
       -> deadline 빠른 순으로 뒤쪽 블록(back)에 배치
       (front+back을 이으면, 그 사이 경계에서 딱 1번만 모델 스위칭이 발생함)
    3) auto 요청(모델 미확정)은 deadline이 급한 것부터 순서대로, "자기 deadline을
       만족시키는 가장 늦은(=덜 방해되는) 삽입 위치"를 찾아서 끼워 넣는다.
       이때 그 위치의 왼쪽/오른쪽 이웃과 같은 모델을 그대로 물려받아서,
       auto 요청 때문에 새로운 스위칭이 추가로 생기지 않도록 한다 (locality-aware).
    PaniniServe는 "정렬"과 "모델 선택"이 서로 얽혀 있어서(어디에 넣을지가 곧
    무슨 모델로 돌지를 정하는 것) 두 개를 따로 안 떼고 이 함수 하나에서 같이 처리한다.
    """
    # 1) front/back 블록 나누기: 지금 로딩된 모델 그대로 쓸 수 있는 요청은 앞에
    front = [r for r in batch if r.model_pref == current_model]
    other_model = "large" if current_model == "small" else "small"
    back = [r for r in batch if r.model_pref == other_model]
    auto_reqs = [r for r in batch if r.model_pref == "auto"]   # 모델이 아직 안 정해진 요청들

    front.sort(key=lambda r: r.deadline)   # 각 블록 내부에서는 deadline 빠른 순 (EDF)
    back.sort(key=lambda r: r.deadline)
    for r in front:
        r.assigned_model = current_model    # front는 전부 현재 모델로 확정
    for r in back:
        r.assigned_model = other_model       # back은 전부 반대쪽 모델로 확정

    queue = front + back   # 이 시점에서 스위칭은 front->back 경계 딱 1번뿐

    # 2) auto 요청들을 deadline이 급한 순서대로 하나씩 큐에 끼워 넣기
    for req in sorted(auto_reqs, key=lambda r: r.deadline):
        best_pos = None
        # 큐의 맨 뒤부터 맨 앞까지 훑으면서, "여기에 넣어도 되는지" 확인
        # (뒤쪽부터 보는 이유: 가능하면 최대한 늦게/방해 안 되게 넣고 싶어서)
        for pos in range(len(queue), -1, -1):
            left = queue[pos - 1].assigned_model if pos > 0 else current_model
            right = queue[pos].assigned_model if pos < len(queue) else None
            # 이웃(왼쪽 우선, 없으면 오른쪽)과 같은 모델로 배정 -> 새 스위칭이 안 생기게 함
            req.assigned_model = left or right or current_model

            # 이 위치에 실제로 넣었다고 가정하고, 큐 맨 앞부터 이 요청까지
            # 순서대로 실행했을 때 걸리는 누적시간을 계산
            t = 0.0
            prev = current_model
            for r in queue[:pos] + [req]:
                t += switch_cost(prev, r.assigned_model)
                t += exec_time(r.assigned_model, r.default_steps)
                prev = r.assigned_model
            # 그 누적시간이 이 요청 자신의 latency_slo(허용시간) 안에 들어오면 이 자리로 확정
            if t <= req.latency_slo:
                best_pos = pos
                break
        if best_pos is None:
            # 어느 자리에 넣어도 latency_slo를 못 맞추면, 일단 맨 뒤에 붙임
            # (뒤쪽 모델을 그대로 물려받아서 최소한 추가 스위칭은 안 나게)
            best_pos = len(queue)
            req.assigned_model = queue[-1].assigned_model if queue else current_model
        queue = queue[:best_pos] + [req] + queue[best_pos:]   # 확정된 위치에 삽입

    return queue


def assign_model(queue: List[Req], policy: str, current_model: str) -> None:
    """Clipper/INFaaS용 모델 배정 (PaniniServe는 order_batch에서 이미 결정했으므로
    여기서는 아무 것도 안 하고 바로 return).
    policy 문자열에 따라 각 요청(r)의 assigned_model을 채워 넣는다."""
    if policy == "panini":
        return  # PaniniServe는 order_batch() 안에서 정렬과 동시에 모델을 이미 정했음 -> 할 일 없음
    elif policy == "clipper_large":
        # Clipper(Large 전용): 요청이 뭘 원했든 무시하고 무조건 large 모델로 고정
        for r in queue:
            r.assigned_model = "large"
    elif policy == "clipper_small":
        # Clipper(Small 전용): 요청이 뭘 원했든 무시하고 무조건 small 모델로 고정
        for r in queue:
            r.assigned_model = "small"
    elif policy == "infaas":
        # INFaaS: 모델 전환 자체는 지원하므로, 큐 재정렬 없이(FIFO 순서 그대로)
        # 각 요청이 원래 원했던 모델(model_pref)을 있는 그대로 배정한다.
        # auto 요청만 예외적으로, latency SLO 계산 때 썼던 calc_model(=large, 보수적 가정)로 배정.
        for r in queue:
            r.assigned_model = r.calc_model if r.model_pref == "auto" else r.model_pref
    else:
        raise ValueError(f"unknown policy: {policy}")


def assign_steps(queue: List[Req], policy: str) -> None:
    """이번 윈도우에서 각 요청이 실제로 실행될 step 수를 초기 배정한다.
    - PaniniServe: 일단 자기 accuracy 레벨의 디폴트(최대) step으로 시작 -> 이후
      윈도우가 넘치면 adapt_steps()가 이 값을 깎는다.
    - Clipper/INFaaS: step을 동적으로 조절하는 기능은 없지만, 그래도 요청이
      원한 accuracy 레벨에 맞춰 그 레벨의 평균(중간) step으로 고정 배정한다
      (BASELINE_STEP_BY_ACCURACY). 한번 정해지면 윈도우 상황과 무관하게 안 바뀐다."""
    if policy == "panini":
        for r in queue:
            r.assigned_steps = r.default_steps      # accuracy 레벨의 최대 step에서 시작
    elif policy in ("clipper_large", "clipper_small", "infaas"):
        for r in queue:
            r.assigned_steps = BASELINE_STEP_BY_ACCURACY[r.accuracy]   # 레벨별 평균값, 고정
    else:
        raise ValueError(f"unknown policy: {policy}")


def adapt_steps(queue: List[Req], current_model: str, max_allowed: float) -> None:
    """PaniniServe 전용 fair-share 동적 step 축소.
    Clipper/INFaaS는 step 조절을 아예 지원 안 하므로 run_scheduler()에서
    policy=="panini"일 때만 이 함수를 호출한다 (그 외 정책은 그냥 건너뜀=pass).

    동작 순서:
      1) 지금 큐를 그대로(디폴트 step으로) 실행하면 총 몇 초 걸리는지 계산
      2) 가용시간(max_allowed)보다 많으면, 초과된 시간(overflow)을
         큐에 있는 요청 수로 균등하게 나눠서(fair-share) 각자 얼마나 시간을
         줄여야 하는지 정함
      3) 그 "줄여야 할 시간"을 그 요청이 쓰는 모델의 step당 시간으로 나눠서
         "몇 step을 깎아야 하는지"로 환산
      4) 단, accuracy 레벨이 보장하는 최소 step(min_steps) 밑으로는 못 내림
         (품질 하한선)
    """
    total = total_window_time(queue, current_model, use_assigned=True)   # 지금 배정대로면 총 몇 초?
    if total <= max_allowed:
        return   # 윈도우 안에 이미 들어오면 축소할 필요 없음

    overflow = total - max_allowed          # 윈도우를 몇 초 초과했는지
    reduce_each = overflow / len(queue)     # 그 초과분을 요청 수만큼 균등하게 나눔 (fair-share)
    for r in queue:
        # "줄여야 할 시간(reduce_each)"을 이 요청 모델의 step당 시간으로 나눠 -> 깎을 step 수
        step_reduction = math.floor(reduce_each / TIME_PER_STEP[r.assigned_model])
        # 원래 step에서 깎되, accuracy 레벨의 최소 step 밑으로는 절대 못 내림
        r.assigned_steps = max(r.default_steps - step_reduction, r.min_steps)


def drop_requests(queue: List[Req], waiting: List[Req], executed: List[Req],
                   current_model: str, max_allowed: float) -> List[Req]:
    """모든 정책(PaniniServe/Clipper/INFaaS) 공통으로 쓰는 drop 로직.
    step을 조절할 수 있는 만큼 다 조절해봐도(PaniniServe) 혹은 애초에 조절이
    불가능해서(Clipper/INFaaS) 여전히 윈도우를 넘치면, deadline이 가장 늦은
    (=상대적으로 가장 급하지 않은) 요청부터 하나씩 통째로 포기(drop)시켜서
    나머지가 윈도우 안에 들어오도록 만든다.

    반환값: drop되고 남은(=이번 윈도우에서 계속 실행할) queue.
    drop된 요청은 이 함수 안에서 곧바로 waiting에서 빼고 executed에 넣어
    최종 결과 리스트에 편입시킨다."""
    total = total_window_time(queue, current_model, use_assigned=True)
    if total <= max_allowed:
        return queue   # 이미 윈도우 안에 들어오면 아무도 안 버려도 됨

    # deadline 내림차순(가장 늦은 것부터) 정렬해서, 하나씩 빼면서 총 시간이
    # max_allowed 이하가 될 때까지 반복
    for r in sorted(queue, key=lambda r: -r.deadline):
        if total <= max_allowed:
            break                      # 이제 윈도우 안에 들어오니 그만 버림
        queue.remove(r)                # 이번 윈도우 실행 대상에서 제외
        if r in waiting:
            waiting.remove(r)          # 대기열에서도 제거 (다음 윈도우로 안 넘어감 = 완전히 버려짐)
        r.dropped, r.status = True, "dropped"
        executed.append(r)             # drop도 최종 결과에는 포함시켜야 metric 계산에서 분모에 들어감
        total = total_window_time(queue, current_model, use_assigned=True)   # 하나 뺐으니 재계산
    return queue


def total_window_time(queue: List[Req], current_model: str, use_assigned: bool = False) -> float:
    """큐에 있는 요청들을 지금 순서대로 쭉 실행했을 때 걸리는 총 시간(초).
    현재 로딩된 모델(current_model)에서 시작해서, 큐를 따라가며 모델이 바뀔
    때마다 switch_cost를 더하고, 각 요청의 실행시간(exec_time)을 누적한다.

    use_assigned=False면 아직 확정 안 된 디폴트(최대) step 기준으로 "만약 이대로
    돌리면 얼마나 걸릴지" 미리 예측하는 용도(예: order_batch의 feasibility 체크)이고,
    use_assigned=True면 실제로 배정이 끝난 assigned_steps 기준으로 계산한다
    (adapt_steps/drop_requests에서 "지금 이 상태로 윈도우 안에 들어오는지" 확인할 때 씀)."""
    t, prev = 0.0, current_model
    for r in queue:
        t += switch_cost(prev, r.assigned_model)   # 직전 모델과 다르면 스위칭 비용 추가
        steps = r.assigned_steps if (use_assigned and r.assigned_steps is not None) else r.default_steps
        t += exec_time(r.assigned_model, steps)     # 이 요청의 순수 실행시간 추가
        prev = r.assigned_model                       # 다음 요청과 비교할 "직전 모델" 갱신
    return t


def run_scheduler(reqs: List[Req], policy: str) -> List[Req]:
    """정책(policy)에 따라 동작하는 통합 윈도우 스케줄링 루프.
    policy: "panini" | "clipper_large" | "clipper_small" | "infaas"

    윈도우/drop 로직(= 시스템 조건)은 4가지 정책이 완전히 동일하게 공유하고,
    정책별로 실제로 갈라지는 지점은 딱 세 군데뿐이다 (아래 (1)(2)(3) 표시):
      (1) Queue 정렬 + 모델 선택   : EDF+locality(panini) vs FIFO(나머지)
      (2) step 배정               : 가변(panini) vs 고정(나머지)
      (3) step 동적 축소 여부      : O(panini) vs X(나머지, adapt_steps 호출 안 함)
    drop 로직(4)과 실제 실행(5)은 모든 정책이 완전히 같은 코드를 공유한다.
    """
    pending = sorted(reqs, key=lambda r: r.arrival)   # 아직 도착 안 한(=대기열에 안 들어간) 요청들
    waiting: List[Req] = []                             # 이미 도착해서 처리를 기다리는 요청들
    current_model = INITIAL_MODEL                        # 지금 GPU에 로딩되어 있다고 가정하는 모델
    window_start, current_time = 0.0, 0.0                 # window_start: 이번 윈도우 시작 시각
                                                            # current_time: GPU가 실제로 다음에 비는 시각
    executed: List[Req] = []                                # met/soft/dropped 전부 포함한 최종 결과

    while len(executed) < len(reqs):   # 전체 요청이 다 처리(실행 or drop)될 때까지 윈도우를 계속 돎
        window_end = window_start + WINDOW_SEC

        # 이번 윈도우가 끝나는 시각까지 도착한 요청들을 pending에서 waiting(대기열)으로 옮김
        arrived = [r for r in pending if r.arrival <= window_end]
        for r in arrived:
            waiting.append(r)
            pending.remove(r)

        if not waiting:
            # 이번 윈도우엔 아직 아무도 도착 안 함 -> 그냥 다음 윈도우로 넘어감
            window_start = window_end
            current_time = max(current_time, window_start)
            continue

        # "윈도우 안에 데드라인이 들어가는 모든 요청을 넣음": deadline<=window_end인 것만 이번 배치 대상
        batch = [r for r in waiting if r.deadline <= window_end]
        if not batch:
            # 아무 요청도 이번 윈도우가 deadline이 아니어도, 손 놓고 있지 말고
            # 대기 중에 가장 급한(가장 이른 deadline) 요청 하나는 처리
            batch = [min(waiting, key=lambda r: r.deadline)]

        # --- (1) Queue 정렬 + 모델 선택 (정책별로 갈라지는 첫 지점) ---
        if policy == "panini":
            queue = order_batch(batch, current_model)   # EDF+locality: 정렬과 모델선택이 한번에 일어남
        else:
            queue = fifo(batch)                            # Clipper/INFaaS: 그냥 도착 순서 그대로
            assign_model(queue, policy, current_model)       # 그 다음 정책별 규칙으로 모델만 따로 배정

        # --- (2) step 배정 (정책별로 갈라지는 두 번째 지점) ---
        assign_steps(queue, policy)   # panini는 디폴트(최대)로 시작, 나머지는 고정값으로 끝

        start_t = max(current_time, window_start)   # 이번 배치를 실제로 시작할 수 있는 시각
        max_allowed = window_end - start_t            # 이번 윈도우 안에서 GPU를 쓸 수 있는 남은 시간
        if max_allowed <= 0:
            # GPU가 이미 윈도우 끝을 넘겨서까지 바빴던 경우 -> 이번 배치는 실행할 시간 자체가 없으니 전부 drop
            for r in queue:
                r.dropped, r.status = True, "dropped"
                if r in waiting:
                    waiting.remove(r)
                executed.append(r)
            window_start = window_end
            continue

        # --- (3) step 동적 축소 (panini만 실제로 동작, 나머지 정책은 애초에 호출 자체를 안 함=no-op) ---
        if policy == "panini":
            adapt_steps(queue, current_model, max_allowed)

        # --- (4) drop (모든 정책이 완전히 동일하게 공유하는 부분) ---
        queue = drop_requests(queue, waiting, executed, current_model, max_allowed)

        # --- (5) 확정된 큐를 실제로 순서대로 실행 (모든 정책이 완전히 동일하게 공유하는 부분) ---
        t, prev = start_t, current_model
        for r in queue:
            sc = switch_cost(prev, r.assigned_model)   # 직전 모델과 다르면 스위칭 비용 발생
            r.switched = sc > 0
            t += sc
            r.start = t                                  # 이 요청의 실제 시작 시각
            t += exec_time(r.assigned_model, r.assigned_steps)
            r.end = t                                     # 이 요청의 실제 종료 시각
            prev = r.assigned_model
            # 종료 시각이 자기 deadline 안이면 met, 넘으면 soft(늦었지만 완료)
            if r.end <= r.deadline:
                r.status = "met"
            else:
                r.status = "soft"
                r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0   # deadline을 몇 % 초과했는지

        if queue:
            current_model = queue[-1].assigned_model   # 다음 윈도우를 위해 "지금 GPU에 로딩된 모델" 갱신
            current_time = t                              # 다음 윈도우를 위해 "GPU가 다음에 비는 시각" 갱신
        for r in queue:
            if r in waiting:
                waiting.remove(r)          # 처리 끝난 요청은 대기열에서 제거
        executed.extend(queue)               # 이번 윈도우에서 실행 완료된 요청들을 최종 결과에 편입
        window_start = window_end             # 다음 윈도우로 이동

    return executed


def run_panini_serve(reqs: List[Req]) -> List[Req]:
    """하위 호환용 래퍼 -- 기존 코드에서 run_panini_serve(reqs)로 부르던 부분이
    그대로 동작하도록 유지."""
    return run_scheduler(reqs, policy="panini")


# =====================================================================
# 7. Metric 계산
# =====================================================================

def compute_metrics(executed: List[Req]) -> Dict[str, float]:
    """전체 요청(drop 포함) 기준으로 4가지 지표를 계산.
    - latency_slo_attainment : deadline 안에 끝난 비율 (%)
    - accuracy_slo_attainment: drop 안 되고 step이 레벨 최소값 이상인 비율 (%)
    - throughput_per_min     : 분당 완료(=drop 안 된) 요청 수
    - dropped_pct            : drop된 비율 (%)
    """
    total = len(executed)
    latency_met = sum(1 for r in executed if r.status == "met")
    accuracy_met = sum(1 for r in executed if r.accuracy_met)
    completed = sum(1 for r in executed if not r.dropped)
    span = max((r.end for r in executed if r.end is not None), default=0.0)
    throughput_per_min = completed / (span / 60.0) if span > 0 else 0.0
    return dict(
        latency_slo_attainment=latency_met / total * 100.0,
        accuracy_slo_attainment=accuracy_met / total * 100.0,
        throughput_per_min=throughput_per_min,
        dropped_pct=sum(1 for r in executed if r.dropped) / total * 100.0,
    )


def run_panini_multi(load_level: str, n_runs: int = 5) -> Dict[str, float]:
    """지정한 부하 수준에서 PaniniServe를 n_runs번 독립 반복 실행하고
    지표 평균±표준편차를 반환."""
    agg = {k: [] for k in ["latency_slo_attainment", "accuracy_slo_attainment",
                            "throughput_per_min", "dropped_pct"]}
    for _ in range(n_runs):
        reqs = generate_requests(load_level, seed=None)   # 매번 새로운 랜덤 (seed 고정 안 함)
        executed = run_panini_serve(reqs)
        metrics = compute_metrics(executed)
        for k, v in metrics.items():
            agg[k].append(v)

    summary = {k: statistics.mean(v) for k, v in agg.items()}
    summary.update({f"{k}_std": statistics.stdev(v) for k, v in agg.items()})
    return summary


# =====================================================================
# 8. Figure 산출
# =====================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Times"],  # 없으면 Times New Roman 메트릭 호환 폰트
    "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
})

LOAD_COLORS = {"low": "#2E5C8A", "medium": "#E8B923", "high": "#C1440E"}


def plot_metric_by_load(results_by_load: Dict[str, Dict[str, float]], metric: str,
                         ylabel: str, title: str, out_path: str, method_names: List[str]):
    """method_names(예: ["Clipper-Large","Clipper-Small","INFaaS","PaniniServe"]) x
    부하 3단계(low/medium/high)로 그룹형 막대 그래프를 그린다.
    results_by_load[load][method] 형태의 dict를 받는다 (팀원 baseline과 합칠 때 이 형식 유지)."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = range(len(method_names))
    width = 0.25
    for i, load in enumerate(["low", "medium", "high"]):
        means = [results_by_load[load][m][metric] for m in method_names]
        stds = [results_by_load[load][m].get(f"{metric}_std", 0.0) for m in method_names]
        positions = [xi + (i - 1) * width for xi in x]
        ax.bar(positions, means, width=width, yerr=stds, capsize=3,
               color=LOAD_COLORS[load], label=load.capitalize())
    ax.set_xticks(list(x))
    ax.set_xticklabels(method_names, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


# =====================================================================
# 9. 실행부 — 지금은 PaniniServe만 3단계 부하로 측정
#    (팀원 Clipper/INFaaS 결과가 오면 results_by_load[load]["Clipper-Large"] 등으로
#     같은 dict에 합쳐서 plot_metric_by_load()를 그대로 재사용하면 됨)
# =====================================================================

if __name__ == "__main__":
    N_RUNS = 5
    results_by_load: Dict[str, Dict[str, Dict[str, float]]] = {}

    for load in ["low", "medium", "high"]:
        print(f"\n### Load level: {load} ({LOAD_LEVELS[load][0]}-{LOAD_LEVELS[load][1]} req/min) ###")
        summary = run_panini_multi(load, n_runs=N_RUNS)
        results_by_load[load] = {"PaniniServe": summary}
        print(f"  PaniniServe: latency SLO {summary['latency_slo_attainment']:.2f}% "
              f"(±{summary['latency_slo_attainment_std']:.2f}), "
              f"accuracy SLO {summary['accuracy_slo_attainment']:.2f}% "
              f"(±{summary['accuracy_slo_attainment_std']:.2f}), "
              f"throughput {summary['throughput_per_min']:.2f}/min, "
              f"dropped {summary['dropped_pct']:.2f}%")

    method_names = ["PaniniServe"]   # 팀원 baseline 합치면 여기에 이름 추가
    plot_metric_by_load(results_by_load, "latency_slo_attainment", "Latency SLO Attainment (%)",
                         "Latency SLO Attainment by Load Level", "graph1_latency_slo.png", method_names)
    plot_metric_by_load(results_by_load, "accuracy_slo_attainment", "Accuracy SLO Attainment (%)",
                         "Accuracy SLO Attainment by Load Level", "graph2_accuracy_slo.png", method_names)
    plot_metric_by_load(results_by_load, "throughput_per_min", "Throughput (req/min)",
                         "Throughput by Load Level", "graph3_throughput.png", method_names)
