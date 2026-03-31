import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
import time
import pandas as pd
import numpy as np

# GPU 가용성 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# 1. 실험 파라미터 설정
# -----------------------
# 노트북/데스크탑 사양에 따라 batch_sizes를 조절하세요. (예: 1, 2, 4, 8)
batch_sizes = [1, 2, 4, 8, 16] 
timesteps = [20, 50]  # 빠른 확인을 위해 스텝 수를 줄여서 테스트 가능
num_runs = 2          # 평균값을 내기 위한 반복 횟수

# 서로 다른 프롬프트 Pool (다중 입력의 원천)
prompt_pool = [
    "a cat sitting on a chair", "a dog in the park", "a futuristic city",
    "a mountain landscape", "a portrait of a woman", "a cyberpunk street",
    "a spaceship in space", "a medieval castle", "a forest with fog",
    "a robot painting", "a sunset beach", "a fantasy dragon"
] * 10 

# -----------------------
# 2. 파이프라인 로드
# -----------------------
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
).to(device)

results = []

print(f"\n[실험 시작] 장치: {device}")
print("Scheduler | Batch | Step | Latency | Throughput | Peak VRAM")
print("-" * 75)

# -----------------------
# 3. 메인 실험 루프
# -----------------------
for step in timesteps:
    for batch in batch_sizes:
        try:
            # (핵심!) 현재 배치 사이즈만큼 서로 다른 프롬프트를 리스트로 묶음
            current_prompts = prompt_pool[:batch]
            
            # --- 눈으로 확인하는 구간 ---
            print(f"\n[입력 확인] 현재 {batch}개의 서로 다른 입력을 묶어서 넣습니다.")
            print(f" > 프롬프트 예시: {current_prompts[0]} ... 외 {batch-1}개")
            # 실제 모델(U-Net) 내부에서는 아래와 같은 4차원 텐서로 계산됩니다.
            print(f" > 예상 모델 입력 텐서 모양: [{batch}, 4, 64, 64] (Batch, Channel, H, W)")
            # --------------------------

            latency_runs = []
            mem_runs = []

            for run in range(num_runs):
                # 캐시 초기화 및 동기화
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                start_time = time.time()

                # 실제 추론 수행 (다중 입력 처리)
                # output_type="latent"로 설정하면 VAE 디코딩 시간을 제외한 순수 확산 연산만 측정 가능합니다.
                _ = pipe(
                    current_prompts, 
                    num_inference_steps=step, 
                    guidance_scale=7.5,
                    output_type="latent" 
                )

                torch.cuda.synchronize()
                end_time = time.time()

                latency_runs.append(end_time - start_time)
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                mem_runs.append(peak_mem)

            # 통계 계산
            avg_latency = np.mean(latency_runs)
            avg_mem = np.mean(mem_runs)
            throughput = batch / avg_latency

            print(f"결과: {batch}배치 | {step}步 | {avg_latency:.2f}s | {throughput:.2f} img/s | {avg_mem:.2f} GB")

            results.append({
                "batch": batch, "step": step, "latency": avg_latency,
                "throughput": throughput, "mem_gb": avg_mem
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"!!! OOM 발생: Batch {batch}는 현재 GPU에서 처리 불가 !!!")
                torch.cuda.empty_cache()
                break # 다음 배치 사이즈로 넘어감
            else:
                raise e

# 데이터 저장
df = pd.DataFrame(results)
df.to_csv("batch_test_results.csv", index=False)
print("\n[완료] 결과가 batch_test_results.csv에 저장되었습니다.")
