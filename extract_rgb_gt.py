#!/usr/bin/env python3
"""从 S2 13波段 clear 提取 RGB 3波段真值"""
import os, sys
import numpy as np
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

RGB = [3, 2, 1]  # Sentinel-2 B4/B3/B2
DATA_ROOT = "/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR"
OUT_ROOT  = "/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR_gt_rgb"

for level in ["60%-80%", "80%-100%"]:
    src_dir = os.path.join(DATA_ROOT, level, "s2")
    out_dir = os.path.join(OUT_ROOT, level, "s2_rgb")
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {level}")
        continue
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in glob(os.path.join(src_dir, "*.tif"))
                    if not f.endswith(".aux.xml")])
    for f in tqdm(files, desc=level):
        with rasterio.open(f) as src:
            data = src.read().astype(np.float32)
            profile = src.profile.copy()
        rgb = data[RGB]
        out_name = os.path.basename(f).replace("_Clear", "").replace("_clear", "")
        out_path = os.path.join(out_dir, out_name)
        profile.update(driver="GTiff", count=3, dtype="float32",
                       compress="lzw", height=rgb.shape[1], width=rgb.shape[2])
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)
    print(f"  {level}: {len(files)} -> {out_dir}/")
print("Done.")
