import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
import lpips
import torchvision.transforms as transforms

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================

VERSION = "flux1-dev"
MODEL_ID = "/mnt/ssd1/jslee/huggingface/hub/FLUX.1-dev-full"

H, W = 1024, 1024  # 생성 및 LPIPS 측정 해상도 통일
FIXED_BATCH_SIZE = 1
TOTAL_IMAGES = 1000
SEED = 42
GUIDANCE_SCALE = 3.5
REFERENCE_STEP = 100  # LPIPS 기준점 스텝

step_sizes = [30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = (
    "/home/jslee/diffusion_exper/batch_exper/dataset/"
    "coco2014/annotation/captions_val2014.json"
)

csv_output_file = f"{base_path}/results/{VERSION}_LPIPS_1024.csv"
ref_images_path = f"{base_path}/ref_{VERSION}_T{REFERENCE_STEP}"

# 1024x1024 원본 해상도 유지를 위해 Resize 레이어를 제거했습니다.
lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ============================================================
# Utils
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco_prompts(path, n):
    print("[INFO] Loading COCO prompts...")
    with open(path, "r") as f:
        data = json.load(f)

    captions = sorted(
        list(set([ann["caption"] for ann in data["annotations"]]))
    )

    return captions[:n]


# ============================================================
# Load Flux
# ============================================================

print(f"[INFO] Loading {MODEL_ID}...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)

prompt_pool = load_coco_prompts(
    coco_annotation_path,
    TOTAL_IMAGES,
)

# ============================================================
# Load LPIPS
# ============================================================

print("[INFO] Loading LPIPS model (VGG)...")
loss_fn_lpips = lpips.LPIPS(net='vgg').cuda()
loss_fn_lpips.eval()

# ============================================================
# Generate Reference Images (T=REFERENCE_STEP)
# ============================================================

if not os.path.exists(ref_images_path) or len(os.listdir(ref_images_path)) < TOTAL_IMAGES:
    print(f"[INFO] Generating reference images (T={REFERENCE_STEP})...")
    os.makedirs(ref_images_path, exist_ok=True)
    seed_everything(SEED)

    with torch.inference_mode():
        for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
            batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
            if not batch_prompts:
                break
            generator = torch.Generator(device="cuda").manual_seed(SEED + i)
            output = pipe(
                prompt=batch_prompts,
                num_inference_steps=REFERENCE_STEP,
                guidance_scale=GUIDANCE_SCALE,
                height=H, width=W,
                generator=generator
            )
            for j, img in enumerate(output.images):
                img.save(os.path.join(ref_images_path, f"{i+j:05d}.png"))
            del output
            torch.cuda.empty_cache()
    print(f"[INFO] Reference images saved: {ref_images_path}\n")
else:
    print(f"[INFO] Reference images already exist: {ref_images_path}")

ref_fnames = sorted(os.listdir(ref_images_path))[:TOTAL_IMAGES]

# ============================================================
# CSV Initialization
# ============================================================

os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)

if not os.path.exists(csv_output_file):
    with open(csv_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LPIPS", "NumImages", "Resolution"])

print(f"{'Steps':<8} | {'LPIPS':<12}")
print("-" * 23)

# ============================================================
# Main Loop (1024px LPIPS Evaluation Only)
# ============================================================

for T in step_sizes:

    seed_everything(SEED)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        total_lpips = 0.0
        count = 0

        with torch.inference_mode():

            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                batch_fnames  = ref_fnames[i : i + FIXED_BATCH_SIZE]

                if not batch_prompts:
                    break

                generator = torch.Generator(device="cuda").manual_seed(SEED + i)

                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    guidance_scale=GUIDANCE_SCALE,
                    height=H, width=W,
                    generator=generator,
                )

                # 1024x1024 크기 그대로 텐서로 변환하여 GPU에 할당
                ref_tensors = torch.stack([
                    lpips_transform(Image.open(os.path.join(ref_images_path, f)).convert('RGB'))
                    for f in batch_fnames
                ]).cuda()

                gen_tensors = torch.stack([
                    lpips_transform(img) for img in output.images
                ]).cuda()

                # LPIPS 계산 (1024 해상도 그대로 연산 수행)
                lpips_distances = loss_fn_lpips(ref_tensors, gen_tensors)
                total_lpips += lpips_distances.sum().item()
                count += len(batch_prompts)

                # 1024 해상도 연산 후 VRAM 확보를 위해 즉시 해제 및 캐시 정리
                del output, ref_tensors, gen_tensors, lpips_distances
                torch.cuda.empty_cache()

        lpips_score = total_lpips / count
        print(f"{T:<8} | {lpips_score:<12.4f}")

        with open(csv_output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([T, lpips_score, TOTAL_IMAGES, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", TOTAL_IMAGES, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", TOTAL_IMAGES, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
