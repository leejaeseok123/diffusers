import sys
import torch
import json
import random
import csv
import os
import gc
import time
import numpy as np
from diffusers import FluxPipeline
from transformers import CLIPProcessor, CLIPModel

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Configuration
# ============================================================

VERSION = "flux1-dev"
MODEL_ID = "/mnt/ssd1/jslee/huggingface/hub/FLUX.1-dev-full"

H, W = 1024, 1024

BATCH_SIZE_QUALITY = 1
BATCH_SIZE_LATENCY = 4

TOTAL_IMAGES_QUALITY = 1000
SEED = 42

GUIDANCE_SCALE = 3.5

step_sizes = [4, 6, 8, 12, 14, 16, 18, 20, 30, 40, 50]

base_path = "/home/jslee/diffusion_exper/batch_exper/fid"

coco_annotation_path = (
    "/home/jslee/diffusion_exper/batch_exper/dataset/"
    "coco2014/annotation/captions_val2014.json"
)

csv_output_file = (
    f"{base_path}/results/{VERSION}_clip_latency.csv"
)

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
    TOTAL_IMAGES_QUALITY,
)

# ============================================================
# Load CLIP
# ============================================================

print("[INFO] Loading CLIP model...")

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).cuda()

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()

# ============================================================
# Warm-up
# ============================================================

print("[INFO] Warm-up...")

with torch.inference_mode():
    _ = pipe(
        prompt_pool[:2],
        num_inference_steps=20,
        guidance_scale=GUIDANCE_SCALE,
        height=H,
        width=W,
    )

torch.cuda.synchronize()

print("[INFO] Warm-up done!\n")

# ============================================================
# CSV Initialization
# ============================================================

os.makedirs(
    os.path.dirname(csv_output_file),
    exist_ok=True,
)

if not os.path.exists(csv_output_file):
    with open(csv_output_file, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Steps",
            "CLIP_Score",
            "User_Latency_sec",
            "NumImages",
            "Resolution",
        ])

print(
    f"{'Steps':<8} | {'CLIP':<10} | {'UserLat(s)':<12}"
)

print("-" * 40)

# ============================================================
# Main Loop
# ============================================================

for T in step_sizes:

    seed_everything(SEED)

    try:

        torch.cuda.empty_cache()
        gc.collect()

        total_clip = 0.0

        with torch.inference_mode():

            for i in range(
                0,
                TOTAL_IMAGES_QUALITY,
                BATCH_SIZE_QUALITY,
            ):

                batch_prompts = prompt_pool[
                    i : i + BATCH_SIZE_QUALITY
                ]

                if not batch_prompts:
                    break

                generator = torch.Generator(
                    device="cuda"
                ).manual_seed(SEED + i)

                output = pipe(
                    prompt=batch_prompts,
                    num_inference_steps=T,
                    guidance_scale=GUIDANCE_SCALE,
                    height=H,
                    width=W,
                    generator=generator,
                )

                # =====================================================
                # CLIP Score
                # =====================================================

                clip_inputs = clip_processor(
                    text=batch_prompts,
                    images=output.images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to("cuda")

                image_embeds = clip_model.get_image_features(
                    pixel_values=clip_inputs["pixel_values"]
                )

                text_embeds = clip_model.get_text_features(
                    input_ids=clip_inputs["input_ids"],
                    attention_mask=clip_inputs["attention_mask"],
                )

                if hasattr(image_embeds, "pooler_output"):
                    image_embeds = image_embeds.pooler_output

                if hasattr(text_embeds, "pooler_output"):
                    text_embeds = text_embeds.pooler_output

                image_embeds = torch.nn.functional.normalize(
                    image_embeds,
                    dim=-1,
                )

                text_embeds = torch.nn.functional.normalize(
                    text_embeds,
                    dim=-1,
                )

                clip_scores = (
                    image_embeds * text_embeds
                ).sum(dim=-1)

                total_clip += clip_scores.sum().item()

                del (
                    output,
                    clip_inputs,
                    image_embeds,
                    text_embeds,
                    clip_scores,
                )

                torch.cuda.empty_cache()

        # =====================================================
        # User Latency
        # =====================================================

        torch.cuda.synchronize()

        start = time.time()

        with torch.inference_mode():

            generator = torch.Generator(
                device="cuda"
            ).manual_seed(SEED)

            _ = pipe(
                prompt=prompt_pool[:BATCH_SIZE_LATENCY],
                num_inference_steps=T,
                guidance_scale=GUIDANCE_SCALE,
                height=H,
                width=W,
                generator=generator,
            )

        torch.cuda.synchronize()

        user_latency = time.time() - start

        clip_score = total_clip / TOTAL_IMAGES_QUALITY

        print(
            f"{T:<8} | "
            f"{clip_score:<10.4f} | "
            f"{user_latency:<12.4f}"
        )

        # =====================================================
        # Save CSV
        # =====================================================

        with open(csv_output_file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                T,
                clip_score,
                user_latency,
                TOTAL_IMAGES_QUALITY,
                f"{H}x{W}",
            ])

    except Exception as e:

        if "out of memory" in str(e).lower():

            print(f"{T:<8} | OOM")

            with open(csv_output_file, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    T,
                    "OOM",
                    "OOM",
                    TOTAL_IMAGES_QUALITY,
                    f"{H}x{W}",
                ])

        else:

            print(f"{T:<8} | ERROR: {e}")

            with open(csv_output_file, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    T,
                    "ERROR",
                    "ERROR",
                    TOTAL_IMAGES_QUALITY,
                    f"{H}x{W}",
                ])

        torch.cuda.empty_cache()
        gc.collect()

print("\n[SUCCESS] Benchmark finished!")
print(os.path.abspath(csv_output_file))               
