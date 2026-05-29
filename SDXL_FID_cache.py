import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from torch.utils.data import Dataset
import torch_fidelity

# 터미널 출력 실시간 확인
sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 1. 설정 및 환경 준비
# -----------------------
VERSION = "SDXL_Base"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
H, W = 1024, 1024
FIXED_BATCH_SIZE = 8  # CLIP 코드의 안정적인 배치 사이즈 패턴 적용
TOTAL_IMAGES = 10000
SEED = 42
CACHE_NAME = f"real_10k_{H}_cache_v1"

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
real_images_path     = f"{base_path}/real_10k_{H}"
csv_output_file      = f"{base_path}/results/{VERSION}_FID.csv"

# 재현성 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# 인메모리 고속 전용 Dataset
# -----------------------
class OptimizedInferenceDataset(Dataset):
    def __init__(self, tensor_pool):
        self.tensor_pool = tensor_pool

    def __len__(self):
        return len(self.tensor_pool)

    def __getitem__(self, idx):
        return self.tensor_pool[idx]

# -----------------------
# 데이터 로드 및 공통 함수
# -----------------------
def seed_everything(target_seed):
    random.seed(target_seed)
    np.random.seed(target_seed)
    torch.manual_seed(target_seed)
    torch.cuda.manual_seed_all(target_seed)

def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = sorted(list(set([ann['caption'] for ann in data['annotations']])))
    return captions[:n]

# Real 데이터셋 전처리
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[*] {VERSION}용 Real 이미지 생성 중 ({H}x{W})...")
    src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"
    os.makedirs(real_images_path, exist_ok=True)
    files = sorted(os.listdir(src))[:TOTAL_IMAGES]
    for i, f in enumerate(files):
        img = Image.open(os.path.join(src, f)).convert('RGB')
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(real_images_path, f))
    print(f"[*] Real 데이터셋 준비 완료!\n")

prompt_pool = load_coco_prompts(coco_annotation_path, TOTAL_IMAGES)

# -----------------------
# 2. 모델 로드 (CLIP 성공 버전과 1:1 완전 일치)
# -----------------------
print(f"[*] Loading SDXL 1.0 ({H}x{W})...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
).to("cuda")

# [CLIP 핵심 조치 1] 다른 부분은 절대 건드리지 않고 오직 VAE만 fp32로 강제 업캐스팅
pipe.vae.to(dtype=torch.float32)

# [CLIP 핵심 조치 2] 최신 API 반영 및 메모리 슬라이싱 활성화
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# ※ 타입 꼬임의 주범이었던 pipe.scheduler 오버라이드 및 pipe.to() 라인을 통째로 제거했습니다.
# 패키지 기본 내장 스케줄러 정렬 상태를 그대로 활용합니다.

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers 가속 활성화")
except ImportError:
    pipe.enable_attention_slicing()
    print("[!] xformers 없음 - Attention Slicing 사용")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# 3. Real 이미지 파일 캐시 등록
# -----------------------
print("[*] Real 이미지 인셉션 특징 추출 및 파일 캐시 등록 중 (최초 1회만)...")
_ = torch_fidelity.calculate_metrics(
    input1=real_images_path, 
    input2=real_images_path, 
    cuda=True, fid=True, verbose=False, save_cpu_ram=True, batch_size=32,
    cache_input1_name=CACHE_NAME,
    cache_input2_name=f"{CACHE_NAME}_dummy"
)
print("[✔] Real 이미지 특징 캐싱 완료!\n")

# -----------------------
# 4. 실험 루프
# -----------------------
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

if not os
