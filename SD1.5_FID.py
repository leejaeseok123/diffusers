import sys
import torch
import time
import json
import random
import csv
import os
import gc
import numpy as np
import torch.nn.functional as F

from diffusers import StableDiffusionPipeline, DDIMScheduler
from pytorch_fid.inception import InceptionV3
from scipy import linalg

sys.stdout.reconfigure(line_buffering=True)

# -----------------------
# 설정
# -----------------------
SEED = 42
device = "cuda"

total_images = 10000
batch_sizes  = [2, 4, 6, 8, 10, 12]
step_sizes   = [4, 6, 8, 12, 16, 20, 30]

H, W = 512, 512

coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
mu_real_path         = "/home/jslee/diffusion_exper/batch_exper/fid/stats/coco2014_mu.npy"
sigma_real_path      = "/home/jslee/diffusion_exper/batch_exper/fid/stats/coco2014_sigma.npy"
csv_output_file      = "/home/jslee/diffusion_exper/batch_exper/fid/results/fid_results.csv"

# -----------------------
# Seed
# -----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -----------------------
# COCO prompt 로드
# -----------------------
def load_coco_prompts(path, n):
    print("[*] Loading COCO prompts...")
    with open(path, 'r') as f:
        data = json.load(f)
    captions = list(set([ann['caption'] for ann in data['annotations']]))
    captions = sorted(captions)
    return captions[:n]

prompt_pool = load_coco_prompts(coco_annotation_path, total_images)

# -----------------------
# 모델 로드
# -----------------------
print("[*] Loading SD v1.5...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
    print("[*] xformers ON")
except:
    print("[!] xformers 없음")

pipe.set_progress_bar_config(disable=True)

# -----------------------
# Inception (FID)
# -----------------------
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
inception = InceptionV3([block_idx]).to(device)
inception.eval()

def get_features(images):
    imgs = []
    for img in images:
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        imgs.append(img)
    imgs = torch.stack(imgs).to(device)
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    imgs = imgs * 2 - 1
    with torch.no_grad():
        pred = inception(imgs)[0]
    return pred.squeeze(3).squeeze(2)  # (B, 2048)

# -----------------------
# Streaming stats (메모리 효율화)
# -----------------------
class RunningStats:
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.S = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x):
        x = x.detach().cpu().numpy().astype(np.float64)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.S += np.outer(delta, delta2)

    def finalize(self):
        if self.n < 2:
            raise ValueError("Not enough samples")
        cov = self.S / (self.n - 1)
        return self.mean, cov

# -----------------------
# FID 계산
# -----------------------
def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(
            (sigma1 + eps * np.eye(len(sigma1))) @
            (sigma2 + eps * np.eye(len(sigma2)))
        )
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

# -----------------------
# 실제 이미지 통계 로드
# -----------------------
print("[*] Loading real image stats...")
mu_real    = np.load(mu_real_path)
sigma_real = np.load(sigma_real_path)
print("[*] 로드 완료!\n")

# -----------------------
# CSV 초기화
# -----------------------
with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Batch", "Steps", "FID"])

print(f"{'Batch':<8} | {'Steps':<8} | {'FID':<10}")
print("-" * 35)

# -----------------------
# 실험 루프
# -----------------------
for T in step_sizes:
    for B in batch_sizes:
        print(f"\n[TEST] Steps={T}, Batch={B}")

        torch.cuda.empty_cache()
        gc.collect()

        stats = RunningStats(2048)

        with torch.inference_mode():
            for i in range(0, total_images, B):
                batch_prompts = prompt_pool[i:i+B]
                if not batch_prompts:
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                images = pipe(
                    batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                ).images

                feats = get_features(images)
                stats.update(feats)

        mu_gen, sigma_gen = stats.finalize()
        fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)

        print(f"{B:<8} | {T:<8} | {fid:<10.2f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([B, T, fid])

print(f"\n[✔] 완료 → {os.path.abspath(csv_output_file)}")
