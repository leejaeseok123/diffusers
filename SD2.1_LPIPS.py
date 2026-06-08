import sys
import torch
import json
import random
import csv
import os
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import lpips
import torchvision.transforms as transforms

sys.stdout.reconfigure(line_buffering=True)

# Experiment Configuration
VERSION = "v2.1"
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
H, W = 768, 768
FIXED_BATCH_SIZE = 32
TOTAL_IMAGES = 1000
SEED = 42

step_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50]

base_path            = "/home/jslee/diffusion_exper/batch_exper/fid"
coco_annotation_path = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/annotation/captions_val2014.json"
coco_image_src       = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/val2014"
real_images_path     = f"{base_path}/real_1k_{H}"
csv_output_file      = f"{base_path}/results/{VERSION}_LPIPS.csv"

# LPIPS 전처리: [-1, 1] 정규화
lpips_transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Helper Functions
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# COCO (image_filename, caption) pair 로드
def load_coco_pairs(path, n):
    print("[INFO] Loading COCO image-caption pairs...")
    with open(path, 'r') as f:
        data = json.load(f)

    id2file = {img['id']: img['file_name'] for img in data['images']}

    seen = set()
    pairs = []
    for ann in sorted(data['annotations'], key=lambda x: x['image_id']):
        img_id = ann['image_id']
        if img_id not in seen:
            seen.add(img_id)
            pairs.append((id2file[img_id], ann['caption']))

    pairs = pairs[:n]
    print(f"[INFO] Loaded {len(pairs)} pairs.")
    return pairs

# Real 이미지 준비
def prepare_real_images(pairs, src_dir, dst_dir, H, W):
    os.makedirs(dst_dir, exist_ok=True)
    existing = set(os.listdir(dst_dir))

    for i, (fname, _) in enumerate(pairs):
        dst_name = f"{i:05d}.png"
        if dst_name in existing:
            continue
        src_path = os.path.join(src_dir, fname)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"[ERROR] Real image not found: {src_path}")
        img = Image.open(src_path).convert('RGB')
        img = img.resize((H, W), resample=Image.LANCZOS)
        img.save(os.path.join(dst_dir, dst_name))

    print(f"[INFO] Real dataset ready: {dst_dir}\n")

# COCO pair 로드
pairs       = load_coco_pairs(coco_annotation_path, TOTAL_IMAGES)
real_fnames = [f"{i:05d}.png" for i in range(len(pairs))]
prompt_pool = [caption for _, caption in pairs]

# Real 이미지 준비
if not os.path.exists(real_images_path) or len(os.listdir(real_images_path)) < TOTAL_IMAGES:
    print(f"[INFO] Preparing real images ({H}x{W})...")
    prepare_real_images(pairs, coco_image_src, real_images_path, H, W)
else:
    print(f"[INFO] Real images already exist: {real_images_path}")

# Model Loading (SD v2.1)
print(f"[INFO] Loading {MODEL_ID}...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

try:
    import xformers
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass

pipe.set_progress_bar_config(disable=True)

# LPIPS 모델 로드
print("[INFO] Loading LPIPS model (VGG)...")
loss_fn_lpips = lpips.LPIPS(net='vgg').cuda()
loss_fn_lpips.eval()

# CSV Initialization
os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
if not os.path.exists(csv_output_file):
    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LPIPS", "NumImages", "Resolution"])

print(f"{'Steps':<8} | {'LPIPS':<12}")
print("-" * 23)

# Main Evaluation Loop
for T in step_sizes:
    seed_everything(SEED)

    try:
        torch.cuda.empty_cache()
        gc.collect()

        total_score = 0.0
        count = 0

        with torch.inference_mode():
            for i in range(0, TOTAL_IMAGES, FIXED_BATCH_SIZE):
                batch_prompts = prompt_pool[i : i + FIXED_BATCH_SIZE]
                batch_fnames  = real_fnames[i : i + FIXED_BATCH_SIZE]
                if not batch_prompts:
                    break

                # 1. 해당 캡션으로 이미지 생성
                generator = torch.Generator(device="cuda").manual_seed(SEED + i)
                output = pipe(
                    batch_prompts,
                    num_inference_steps=T,
                    height=H, width=W,
                    generator=generator
                )

                # 2. 대응되는 real 이미지 로드
                real_tensors = torch.stack([
                    lpips_transform(Image.open(os.path.join(real_images_path, f)).convert('RGB'))
                    for f in batch_fnames
                ]).cuda()

                # 3. generated → 텐서 변환 (저장 없이 바로)
                gen_tensors = torch.stack([
                    lpips_transform(img) for img in output.images
                ]).cuda()

                # 4. LPIPS 계산
                d = loss_fn_lpips(real_tensors, gen_tensors)  # (B, 1, 1, 1)
                total_score += d.sum().item()
                count += len(batch_prompts)

                # 5. 즉시 메모리 해제
                del real_tensors, gen_tensors, d, output
                torch.cuda.empty_cache()

        lpips_score = total_score / count

        print(f"{T:<8} | {lpips_score:<12.4f}")

        with open(csv_output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([T, lpips_score, TOTAL_IMAGES, f"{H}x{W}"])

    except Exception as e:
        if "out of memory" in str(e).lower():
            print(f"{T:<8} | OOM")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "OOM", TOTAL_IMAGES, f"{H}x{W}"])
        else:
            print(f"{T:<8} | ERROR: {e}")
            with open(csv_output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([T, "ERROR", TOTAL_IMAGES, f"{H}x{W}"])
        torch.cuda.empty_cache()
        gc.collect()

print(f"\n[SUCCESS] Benchmark finished -> {os.path.abspath(csv_output_file)}")
