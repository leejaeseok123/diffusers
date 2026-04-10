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

# 터미널에서 확인한 사용자님의 경로 반영
# 상대 경로 대신 절대 경로를 쓰면 훨씬 안전합니다.
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file = "performance_results_final.csv"

# 실험 변수 설정 (1, 6, 11, ..., 96)
batch_sizes = list(range(1, 101, 5))
step_sizes = [20, 50, 100]
num_runs = 3
total_images_to_gen = 10000

def load_coco_prompts(json_path, num_samples):
    print(f"[*] Loading COCO prompts from {json_path}...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"에러: 파일을 찾을 수 없습니다. 경로를 확인하세요: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 캡션 추출 및 중복 제거
    all_captions = [ann['caption'] for ann in data['annotations']]
    unique_captions = list(set(all_captions))
    
    if len(unique_captions) < num_samples:
        print(f"[!] Warning: 고유 캡션 수가 요청보다 적어 {len(unique_captions)}개만 사용합니다.")
        num_samples = len(unique_captions)
        
    return random.sample(unique_captions, num_samples)

# -----------------------
# 2. 모델 로드 및 최적화 (RTX 6000 Ada 전용)
# -----------------------
print("[*] Loading Stable Diffusion Model (v1.5)...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None  # 순수 속도 측정을 위해 안전 검사기 비활성화
).to(device)

# DDIM 스케줄러 적용 및 성능 옵션 활성화
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()

# torch.compile: Ada 아키텍처의 연산 효율을 극대화
try:
    print("[*] torch.compile 시작 (최적화 분석을 위해 수 분이 소요될 수 있습니다)...")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
except Exception as e:
    print(f"[!] torch.compile 건너뜀: {e}")

pipe.set_progress_bar_config(disable=True)
prompt_pool = load_coco_prompts(coco_annotation_path, total_images_to_gen)

# -----------------------
# 3. Warm-up (최초 1회 예열)
# -----------------------
print("\n[*] GPU Warm-up 및 컴파일 캐싱 중...")
with torch.inference_mode():
    for _ in range(3):
        # 작은 배치로 미리 연산하여 CUDA 커널을 준비시킵니다.
        _ = pipe(prompt_pool[:2], num_inference_steps=20)
torch.cuda.synchronize()
print("[*] Warm-up 완료!\n")

# -----------------------
# 4. 메인 실험 루프 (OOM 대응)
# -----------------------
# CSV 결과 파일 초기화
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Run", "Batch", "Steps", "Total_Time_s", "Latency_s_per_img", "Throughput_img_s", "Peak_Mem_GB"])

print(f"{'Run':<5} | {'Batch':<6} | {'Steps':<6} | {'Latency(s/img)':<15} | {'Throughput':<12} | {'PeakMem':<8}")
print("-" * 85)

for run in range(1, num_runs + 1):
    for B in batch_sizes:
        for T in step_sizes:
            try:
                # 하드웨어 상태 초기화
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()

                torch.cuda.synchronize()
                start_time = time.time()

                with torch.inference_mode():
                    # 전체 10,000장을 지정된 배치 사이즈로 나누어 생성
                    for i in range(0, total_images_to_gen, B):
                        current_batch = prompt_pool[i : i + B]
                        if not current_batch: break
                        # 생성 (이미지 결과는 메모리 주소만 참조 후 즉시 버림)
                        _ = pipe(current_batch, num_inference_steps=T).images

                torch.cuda.synchronize()
                end_time = time.time()

                # 결과 데이터 가공
                total_elapsed = end_time - start_time
                latency_per_img = total_elapsed / total_images_to_gen # 1장당 소요 시간
                throughput = total_images_to_gen / total_elapsed     # 1초당 생성 매수
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3

                print(f"{run:<5} | {B:<6} | {T:<6} | {latency_per_img:<15.4f} | {throughput:<12.2f} | {peak_mem:<8.2f}")
                
                # 실시간 결과 저장
                with open(csv_output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, B, T, total_elapsed, latency_per_img, throughput, peak_mem])

            except RuntimeError as e:
                # OOM 발생 시 예외 처리
                if "out of memory" in str(e).lower():
                    print(f"{run:<5} | {B:<6} | {T:<6} | {'OOM':<15} | {'OOM':<12} | {'OOM':<8}")
                    with open(csv_output_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run, B, T, "OOM", "OOM", "OOM", "OOM"])
                    
                    # 메모리 청소 후 계속 진행
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e

print(f"\n[!] 모든 실험이 성공적으로 완료되었습니다.")
print(f"[!] 결과 데이터: {os.path.abspath(csv_output_file)}")
