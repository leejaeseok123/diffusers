import os
import shutil

src = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/val2014"
dst = "/home/jslee/diffusion_exper/batch_exper/dataset/coco2014/val2014/real_10k"

os.makedirs(dst, exist_ok=True)

files = sorted(os.listdir(src))[:10000]
for i, f in enumerate(files):
    shutil.copy(os.path.join(src, f), dst)
    if i % 1000 == 0:
        print(f"[*] {i}/10000 복사 중...")

print(f"[✔] 완료! {dst}")
