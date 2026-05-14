import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np

from diffusers import StableDiffusionXLPipeline
from transformers import CLIPProcessor, CLIPModel

# 터미널 출력 실시간 확인
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 1. 설정 및 환경 준비
# -----------------------
SEED = 42
device = "cuda"

total_images = 1000
batch_size   = 16  # 48GB VRAM 기준 16이 적당하며, OOM 발생 시 8로 줄이세요.
step_sizes   = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]
H, W = 1024, 1024

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/SDXL_clip_results.csv"

# 재현성 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# 2. 데이터 로드
# -----------------------
def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    if not os.path.exists(path):
        print(f"[!] 경로 없음: {path}. 기본 프롬프트를 사용합니다.")
        return ["a high quality professional photo"] * n
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 3. 모델 로드 및 타입 패치 (RuntimeError 해결 핵심)
# -----------------------
print(f"[*] Loading SDXL 1.0 ({H}x{W})...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to(device)

# [해결책 1] VAE 가중치를 fp32로 변환
pipe.vae.to(dtype=torch.float32)

# [해결책 2] 파이프라인 decode 함수 패치
# UNet의 fp16 결과물(latents)을 VAE에 넣기 직전에 fp32로 변환하여 타입 충돌 방지
original_decode = pipe.decode
def forced_fp32_decode(latents, *args, **kwargs):
    return original_decode(latents.to(dtype=torch.float32), *args, **kwargs)
pipe.decode = forced_fp32_decode

# 최신 API 최적화 설정
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers 가속 활성화")
except ImportError:
    pipe.enable_attention_slicing()

pipe.set_progress_bar_config(disable=True)

# -----------------------
# 4. CLIP 모델 로드
# -----------------------
print("[*] Loading CLIP model...")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
print("[*] CLIP 로드 완료!\n")

def calc_clip_score(images, prompts):
    inputs = clip_processor(
        text=prompts, images=images,
        return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs    = clip_model(**inputs)
        img_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_embeds = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_embeds * txt_embeds).sum(dim=-1)
    return scores.cpu().float().tolist()

# Warm-up
print("[*] Warm-up 중 (VAE Decoding 포함)...")
with torch.inference_mode():
    _ = pipe(prompt_pool[:2], num_inference_steps=10, height=H, width=W).images
torch.cuda.synchronize()
print("[*] Warm-up 완료!\n")

# -----------------------
# 5. 실험 루프
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "CLIP_Score"])

print(f"{'Steps':<8} | {'CLIP_Score':<12}")
print("-" * 25)

for T in step_sizes:
    print(f"\n[TEST] Steps={T}, Batch={batch_size}(고정)")
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        all_scores = []
        with torch.inference_mode():
            for i in range(0, total_images, batch_size):
                batch_prompts = prompt_pool[i : i + batch_size]
                if not batch_prompts: break
                
                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                
                # 이미지 생성
                output = pipe(
                    prompt=batch_prompts, 
                    num_inference_steps=T,
                    height=H, 
                    width=W, 
                    generator=generator
                )
                images = output.images
                
                # CLIP 점수 계산
                batch_scores = calc_clip_score(images, batch_prompts)
                all_scores.extend(batch_scores)
                
                if (i + batch_size) % 80 == 0:
                    print(f"  → {i + batch_size}/{total_images} 이미지 처리 중...")

        clip_score = float(np.mean(all_scores))
        print(f"[*] 결과 → Steps {T}: {clip_score:.4f}")
        
        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, clip_score])
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  → [!] OOM! T={T} 스킵")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM"])
        else:
            print(f"  → [!] 에러: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR"])
        
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[✔] 실험 완료! 결과: {csv_output_file}")
