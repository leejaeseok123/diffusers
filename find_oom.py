import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# OOM 탐색 설정
# -----------------------
batch_sizes = [48, 50, 52, 54, 56, 58, 60, 62, 64]
timestep = 100
num_runs = 1   # OOM 탐색은 1번이면 충분

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
# Scheduler (하나만)
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

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM 발생 at batch={batch}")
            torch.cuda.empty_cache()
            break
        else:
            raise e
