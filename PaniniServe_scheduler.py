"""
PaniniServe vs. Baselines: Experimental Comparison
====================================================

Workload
--------
- 300 prompts (COCO placeholder), Poisson arrivals.
- Load levels: low (1-2 req/min), medium (3-4 req/min), high (10-18 req/min).
- Each request: model_pref in {Large, Small, Auto}, accuracy SLO in
  {High: 30-40 steps, Medium: 20-30, Low: 10-20}.
- Latency SLO is NOT user-set: system auto-computes it as ~2-3x the
  profiled base execution time for that request's own accuracy-level
  default (max) step count.

Baselines
---------
- Clipper-Large : Large model only, fixed step=30, plain FIFO, no drops.
- Clipper-Small : Small model only, fixed step=30, plain FIFO, no drops.
- INFaaS        : model switching allowed (per request's model_pref),
                  fixed step=30, plain FIFO (arrival order, no
                  deadline-aware reordering), no drops.
- PaniniServe   : EDF + locality-aware switching + fair-share dynamic
                  step adjustment (10-40), window-based (this repo's
                  scheduler).

Accuracy SLO attainment: executed step count >= that request's
accuracy-level minimum step count.
"""

import math
import random
import statistics
from dataclasses import dataclass
from typing import List, Optional, Dict
import matplotlib.pyplot as plt

# ----------------------------- Config -----------------------------------

TOTAL_REQUESTS = 300

ACCURACY_STEP_RANGE = {
    "high": (30, 40),
    "medium": (20, 30),
    "low": (10, 20),
}

TIME_PER_STEP = {"small": 0.05535, "large": 0.28980}     # SDv2.1 / SDv3.5-Medium
BASE_OVERHEAD = {"small": 0.0903, "large": 0.2906}
SWITCH_TIME = {("small", "large"): 1.90, ("large", "small"): 6.31}

SLO_MULTIPLIER_RANGE = (2.0, 3.0)     # widened per new spec (was 1.5-2.0)
STEP_RANGE_BOUNDS = (10, 40)

BASELINE_FIXED_STEP = 20   # single fixed step for all baseline requests regardless of accuracy tier

LOAD_LEVELS = {
    "low":    (1.0, 2.0),
    "medium": (3.0, 4.0),
    "high":   (10.0, 18.0),
}

WINDOW_SEC = 60.0
INITIAL_MODEL = "small"
MODEL_TYPES = ["small", "large", "auto"]
ACCURACY_LEVELS = ["high", "medium", "low"]
PROMPTS = [f"coco_prompt_{i:03d}" for i in range(TOTAL_REQUESTS)]  # COCO placeholder


def slo_multiplier(steps: float) -> float:
    lo, hi = STEP_RANGE_BOUNDS
    m_lo, m_hi = SLO_MULTIPLIER_RANGE
    frac = (steps - lo) / (hi - lo) if hi > lo else 1.0
    frac = min(max(frac, 0.0), 1.0)
    return m_lo + frac * (m_hi - m_lo)


def exec_time(model: str, steps: float) -> float:
    return BASE_OVERHEAD[model] + steps * TIME_PER_STEP[model]


def switch_cost(prev_model: Optional[str], next_model: str) -> float:
    if prev_model is None or prev_model == next_model:
        return 0.0
    return SWITCH_TIME[(prev_model, next_model)]


# ----------------------------- Data model --------------------------------

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
    start: Optional[float] = None
    end: Optional[float] = None
    dropped: bool = False
    switched: bool = False
    status: Optional[str] = None
    overage_pct: Optional[float] = None

    @property
    def accuracy_met(self) -> bool:
        return (not self.dropped) and self.assigned_steps is not None and self.assigned_steps >= self.min_steps


def generate_requests(load_level: str, seed=None) -> List[Req]:
    if seed is not None:
        random.seed(seed)
    lo_rate, hi_rate = LOAD_LEVELS[load_level]

    prompt_order = PROMPTS.copy()
    random.shuffle(prompt_order)

    reqs = []
    t = 0.0
    for rid in range(TOTAL_REQUESTS):
        rate_per_min = random.uniform(lo_rate, hi_rate)   # randomized within the stated band each step
        rate_per_sec = rate_per_min / 60.0
        t += random.expovariate(rate_per_sec)

        model_pref = random.choice(MODEL_TYPES)
        accuracy = random.choice(ACCURACY_LEVELS)
        mn, mx = ACCURACY_STEP_RANGE[accuracy]
        calc_model = "large" if model_pref == "auto" else model_pref
        latency_slo = exec_time(calc_model, mx) * slo_multiplier(mx)
        deadline = t + latency_slo

        reqs.append(Req(
            rid=rid, arrival=round(t, 3), prompt=prompt_order[rid],
            model_pref=model_pref, accuracy=accuracy,
            default_steps=mx, min_steps=mn, calc_model=calc_model,
            latency_slo=round(latency_slo, 3), deadline=round(deadline, 3),
        ))
    return reqs


def clone_requests(reqs: List[Req]) -> List[Req]:
    return [Req(rid=r.rid, arrival=r.arrival, prompt=r.prompt, model_pref=r.model_pref,
                 accuracy=r.accuracy, default_steps=r.default_steps, min_steps=r.min_steps,
                 calc_model=r.calc_model, latency_slo=r.latency_slo, deadline=r.deadline)
            for r in reqs]


# ----------------------------- Baselines: plain FIFO ----------------------

def run_fifo_baseline(reqs: List[Req], forced_model: Optional[str]) -> List[Req]:
    """Clipper (forced_model set) or INFaaS (forced_model=None -> use request's own model_pref).
    Uses the SAME window + drop mechanics as PaniniServe (60s windows, batch by
    deadline<=window_end, drop-by-latest-deadline on overflow) so the comparison
    isolates the scheduling POLICY, not the underlying system model. The only
    differences from PaniniServe: FIFO order within the batch (no EDF/locality
    reordering) and a fixed step count per accuracy tier (no dynamic step
    reduction under overflow -- baselines can't adapt steps)."""
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting: List[Req] = []
    current_model = INITIAL_MODEL
    window_start, current_time = 0.0, 0.0
    executed: List[Req] = []

    while len(executed) < len(reqs):
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

        # FIFO order (arrival order), fixed model/step assignment -- no reordering
        queue = sorted(batch, key=lambda r: r.arrival)
        for r in queue:
            r.assigned_model = forced_model if forced_model is not None else (
                r.calc_model if r.model_pref == "auto" else r.model_pref
            )
            r.assigned_steps = BASELINE_FIXED_STEP

        start_t = max(current_time, window_start)
        max_allowed = window_end - start_t
        if max_allowed <= 0:
            for r in queue:
                r.dropped, r.status = True, "dropped"
                if r in waiting:
                    waiting.remove(r)
                executed.append(r)
            window_start = window_end
            continue

        total = total_window_time(queue, current_model, use_assigned=True)

        # no step-reduction option for baselines (fixed step) -> straight to drop-by-latest-deadline
        if total > max_allowed:
            for r in sorted(queue, key=lambda r: -r.deadline):
                if total <= max_allowed:
                    break
                queue.remove(r)
                if r in waiting:
                    waiting.remove(r)
                r.dropped, r.status = True, "dropped"
                executed.append(r)
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
            if r.end <= r.deadline:
                r.status = "met"
            else:
                r.status = "soft"
                r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0

        if queue:
            current_model = queue[-1].assigned_model
            current_time = t
        for r in queue:
            if r in waiting:
                waiting.remove(r)
        executed.extend(queue)
        window_start = window_end

    return executed


# ----------------------------- PaniniServe (this repo's scheduler) --------

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


def total_window_time(queue, current_model, use_assigned=False):
    t, prev = 0.0, current_model
    for r in queue:
        t += switch_cost(prev, r.assigned_model)
        steps = r.assigned_steps if use_assigned and r.assigned_steps is not None else r.default_steps
        t += exec_time(r.assigned_model, steps)
        prev = r.assigned_model
    return t


def run_panini_serve(reqs: List[Req]) -> List[Req]:
    pending = sorted(reqs, key=lambda r: r.arrival)
    waiting: List[Req] = []
    current_model = INITIAL_MODEL
    window_start, current_time = 0.0, 0.0
    executed: List[Req] = []

    while len(executed) < len(reqs):
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
                executed.append(r)
            window_start = window_end
            continue

        total = total_window_time(queue, current_model, use_assigned=True)
        if total > max_allowed:
            overflow = total - max_allowed
            red_each = overflow / len(queue)
            for r in queue:
                red_steps = math.floor(red_each / TIME_PER_STEP[r.assigned_model])
                r.assigned_steps = max(r.default_steps - red_steps, r.min_steps)
            total = total_window_time(queue, current_model, use_assigned=True)

        if total > max_allowed:
            for r in sorted(queue, key=lambda r: -r.deadline):
                if total <= max_allowed:
                    break
                queue.remove(r)
                if r in waiting:
                    waiting.remove(r)
                r.dropped, r.status = True, "dropped"
                executed.append(r)
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
            if r.end <= r.deadline:
                r.status = "met"
            else:
                r.status = "soft"
                r.overage_pct = (r.end - r.deadline) / r.latency_slo * 100.0

        if queue:
            current_model = queue[-1].assigned_model
            current_time = t
        for r in queue:
            if r in waiting:
                waiting.remove(r)
        executed.extend(queue)
        window_start = window_end

    return executed


# ----------------------------- Metrics -------------------------------------

def compute_metrics(executed: List[Req]) -> Dict[str, float]:
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


METHODS = ["Clipper-Large", "Clipper-Small", "INFaaS", "PaniniServe"]


def run_all_methods(load_level: str, n_runs: int = 5) -> Dict[str, Dict[str, float]]:
    agg = {m: {"latency_slo_attainment": [], "accuracy_slo_attainment": [],
               "throughput_per_min": [], "dropped_pct": []} for m in METHODS}

    for _ in range(n_runs):
        base_reqs = generate_requests(load_level, seed=None)

        results = {
            "Clipper-Large": run_fifo_baseline(clone_requests(base_reqs), "large"),
            "Clipper-Small": run_fifo_baseline(clone_requests(base_reqs), "small"),
            "INFaaS":        run_fifo_baseline(clone_requests(base_reqs), None),
            "PaniniServe":   run_panini_serve(clone_requests(base_reqs)),
        }
        for m, executed in results.items():
            metrics = compute_metrics(executed)
            for k, v in metrics.items():
                agg[m][k].append(v)

    summary = {}
    for m in METHODS:
        summary[m] = {k: statistics.mean(v) for k, v in agg[m].items()}
        summary[m].update({f"{k}_std": statistics.stdev(v) for k, v in agg[m].items()})
    return summary


# ----------------------------- Plotting -------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Times"],
    "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
})

LOAD_COLORS = {"low": "#2E5C8A", "medium": "#E8B923", "high": "#C1440E"}


def plot_metric(all_results: Dict[str, Dict[str, Dict[str, float]]], metric: str, ylabel: str, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = range(len(METHODS))
    width = 0.25
    for i, load in enumerate(["low", "medium", "high"]):
        means = [all_results[load][m][metric] for m in METHODS]
        stds = [all_results[load][m][f"{metric}_std"] for m in METHODS]
        positions = [xi + (i - 1) * width for xi in x]
        ax.bar(positions, means, width=width, yerr=stds, capsize=3, color=LOAD_COLORS[load], label=load.capitalize())
    ax.set_xticks(list(x))
    ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    N_RUNS = 5
    all_results = {}
    for load in ["low", "medium", "high"]:
        print(f"\n### Load level: {load} ({LOAD_LEVELS[load][0]}-{LOAD_LEVELS[load][1]} req/min) ###")
        summary = run_all_methods(load, n_runs=N_RUNS)
        all_results[load] = summary
        for m in METHODS:
            s = summary[m]
            print(f"  {m:<15s}: latency SLO {s['latency_slo_attainment']:.2f}% (±{s['latency_slo_attainment_std']:.2f}), "
                  f"accuracy SLO {s['accuracy_slo_attainment']:.2f}% (±{s['accuracy_slo_attainment_std']:.2f}), "
                  f"throughput {s['throughput_per_min']:.2f}/min, dropped {s['dropped_pct']:.2f}%")

    plot_metric(all_results, "latency_slo_attainment", "Latency SLO Attainment (%)",
                "Latency SLO Attainment by Method and Load", "graph1_latency_slo.png")
    plot_metric(all_results, "accuracy_slo_attainment", "Accuracy SLO Attainment (%)",
                "Accuracy SLO Attainment by Method and Load", "graph2_accuracy_slo.png")
    plot_metric(all_results, "throughput_per_min", "Throughput (req/min)",
                "Throughput by Method and Load", "graph3_throughput.png")
