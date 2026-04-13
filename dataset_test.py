import torch
import time
import json
import random
import csv
import os
import gc
from diffusers import StableDiffusionPipeline, DDIMScheduler

# -----------------------
# 1. 환경 설정 및 데이터 로드
# -----------------------
device = "cuda"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "performance_results_final.csv"

# 실험 변수 설정
batch_sizes = list(range(1, 131, 8))  # 1, 6, 11, ..., 96
step_sizes = [10, 20, 30, 50, 75, 100, 150]
num_runs = 3
total_images_to_gen = 10000

def load_coco_prompts(json_path, num_samples):
    print(f"[*] Loading COCO prompts from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_captions = [ann['caption'] for ann in data['annotations']]
    unique_captions = list(set(all_captions))
    return random.sample(unique_captions, num_samples)

# -----------------------
# 2. 모델 로드 (컴파일 OFF)
# -----------------------
print("[*] Loading Stable Diffusion Model (v1.5)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.set_progress_bar_config(disable=True)

prompt_pool = load_coco_prompts(coco_annotation_path, total_images_to_gen)

# -----------------------
# 3. Warm-up (최소 1회)
# -----------------------
print("\n[*] GPU Warm-up 1회 진행 중...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=20)
torch.cuda.synchronize()
print("[*] 준비 완료! 본 실험을 시작합니다.\n")

# -----------------------
# 4. 메인 실험 루프 (OOM 시 break 적용)
# -----------------------
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Batch", "Steps", "Total_Time_s", "Latency_s_per_img", "Throughput_img_s", "Peak_Mem_GB"])

print(f"{'Run':<5} | {'Batch':<6} | {'Steps':<6} | {'Latency(s/img)':<15} | {'Throughput':<12} | {'PeakMem':<8}")
print("-" * 85)

for run in range(1, num_runs + 1):
    for T in step_sizes: # Steps를 먼저 고정하고 Batch를 키우는 것이 효율적입니다.
        for B in batch_sizes:
            try:
                # 하드웨어 초기화
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()

                torch.cuda.synchronize()
                start_time = time.time()

                with torch.inference_mode():
                    for i in range(0, total_images_to_gen, B):
                        current_batch = prompt_pool[i : i + B]
                        if not current_batch: break
                        _ = pipe(current_batch, num_inference_steps=T).images

                torch.cuda.synchronize()
                end_time = time.time()

                # 지표 계산
                total_elapsed = end_time - start_time
                latency_per_img = total_elapsed / total_images_to_gen
                throughput = total_images_to_gen / total_elapsed
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                print(f"{run:<5} | {B:<6} | {T:<6} | {latency_per_img:<15.4f} | {throughput:<12.2f} | {peak_mem:<8.2f}")
                
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, total_elapsed, latency_per_img, throughput, peak_mem])

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{run:<5} | {B:<6} | {T:<6} | {'OOM (Skipping larger batches)':<35}")
                    
                    # CSV에 OOM 기록
                    with open(csv_output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM"])
                    
                    # 메모리 청소
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # 중요: 현재 T에서 OOM이 났으므로, 더 큰 B는 시도하지 않고 다음 T로 넘어감
                    break 
                else:
                    raise e

print(f"\n[!] 실험이 모두 완료되었습니다. 결과 파일: {os.path.abspath(csv_output_file)}")
