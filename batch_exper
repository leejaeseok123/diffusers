import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import time
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 실험 범위
# -----------------------
batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96]
timesteps = [10, 20, 30, 50, 75, 100, 150, 200]

num_runs = 3

# -----------------------
# 서로 다른 프롬프트 pool
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
] * 10   # batch 96 커버

# -----------------------
# Scheduler 설정
# -----------------------
schedulers = {
    "DDIM": DDIMScheduler,
    "DDPM": DDPMScheduler
}

results = []

# GPU 전체 메모리
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU Total Memory: {total_mem:.2f} GB")

print("Scheduler | Batch | Step | Latency | Throughput | PeakMem")
print("------------------------------------------------------------------")

# -----------------------
# 실험 loop
# -----------------------
for scheduler_name, scheduler_class in schedulers.items():

    print(f"\n===== {scheduler_name} START =====")

    scheduler = scheduler_class.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16
    ).to(device)

    for step in timesteps:
        for batch in batch_sizes:
            try:
                latency_runs = []
                peak_mem_runs = []

                for _ in range(num_runs):

                    prompts = prompt_pool[:batch]

                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                    start = time.time()

                    _ = pipe(
                        prompts,
                        num_inference_steps=step,
                        guidance_scale=7.5
                    ).images

                    torch.cuda.synchronize()
                    end = time.time()

                    latency_runs.append(end - start)

                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    peak_mem_runs.append(peak)

                avg_latency = np.mean(latency_runs)
                avg_peak = np.mean(peak_mem_runs)

                throughput = batch / avg_latency
                util_ratio = avg_peak / total_mem

                print(
                    f"{scheduler_name} | {batch} | {step} | "
                    f"{avg_latency:.3f}s | {throughput:.2f} img/s | "
                    f"{avg_peak:.2f} GB"
                )

                results.append({
                    "scheduler": scheduler_name,
                    "batch": batch,
                    "step": step,
                    "latency": avg_latency,
                    "throughput": throughput,
                    "gpu_peak_GB": avg_peak,
                    "gpu_total_GB": total_mem,
                    "gpu_util_ratio": util_ratio
                })

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at {scheduler_name}, batch={batch}, step={step}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

# -----------------------
# CSV 저장
# -----------------------
df = pd.DataFrame(results)
df.to_csv("edge_diffusion_batch_results.csv", index=False)

print("\nSaved to edge_diffusion_batch_results.csv")
