#!/usr/bin/env python3
"""TIF 预测结果 → 2/98% 截断 PNG"""
import os, sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

SRC = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/tif"
OUT = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/rgb_png"

for level in ["0-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]:
    src_dir = os.path.join(SRC, level, "pred")
    out_dir = os.path.join(OUT, level)
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {level}")
        continue
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(src_dir, "*.tif")))
    for f in tqdm(files, desc=level):
        with rasterio.open(f) as s:
            img = s.read().astype(np.float32)  # [3, H, W]
        for c in range(3):
            lo, hi = np.percentile(img[c], 2), np.percentile(img[c], 98)
            img[c] = np.clip((img[c] - lo) / (hi - lo + 1e-8), 0, 1)
        img = (np.transpose(img, (1, 2, 0)) * 255).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, os.path.basename(f).replace(".tif", ".png"))
        Image.fromarray(img).save(out_path)
    print(f"  {level}: {len(files)} -> {out_dir}/")
print("Done.")
