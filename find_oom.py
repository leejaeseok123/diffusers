import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import time
import numpy as np
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# OOM 탐색 설정 (범위 확장)
# -----------------------
batch_sizes = list(range(40, 97, 2))   # 40~96 step 2
timestep = 100
num_runs = 1

# -----------------------
# CSV 저장
# -----------------------
csv_file = "oom_search_results.csv"
results = []

# -----------------------
# 프롬프트
# -----------------------
prompt_pool = [
    "a cat sitting on a chair",
    "a dog in the park",
    "a futuristic city",
    "a mountain landscape",
    "a portrait of a woman",
    "a cyberpunk street",
    "a spaceship in space",
    "a medieval castle",
    "a forest with fog",
    "a robot painting",
    "a sunset beach",
    "a fantasy dragon",
    "a snowy village",
    "a racing car",
    "a watercolor painting",
    "a sci-fi soldier",
    "a lake reflection",
    "a desert landscape",
    "a macro flower",
    "a night skyline"
] * 10

# -----------------------
# Scheduler
# -----------------------
scheduler = DDIMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)

total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU Total Memory: {total_mem:.2f} GB")

print("Batch | Latency | Throughput | PeakMem")
print("-----------------------------------------")

# -----------------------
# OOM 탐색 loop
# -----------------------
for batch in batch_sizes:
    try:
        prompts = prompt_pool[:batch]

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()

        _ = pipe(
            prompts,
            num_inference_steps=timestep,
            guidance_scale=7.5
        ).images

        torch.cuda.synchronize()
        end = time.time()

        latency = end - start
        peak = torch.cuda.max_memory_allocated() / 1024**3
        throughput = batch / latency

        print(
            f"{batch} | {latency:.3f}s | "
            f"{throughput:.2f} img/s | {peak:.2f} GB"
        )

        results.append([batch, latency, throughput, peak])

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM 발생 at batch={batch}")
            results.append([batch, "OOM", "OOM", "OOM"])
            break
        else:
            raise e

# -----------------------
# CSV 저장
# -----------------------
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["batch_size", "latency", "throughput", "peak_memory_GB"])
    writer.writerows(results)

print(f"\n결과 저장 완료 → {csv_file}")
