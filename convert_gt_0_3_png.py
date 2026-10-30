#!/usr/bin/env python3
"""GT TIF → PNG: [0,10000]归一化 → ×3 → clip[0,1] → uint8"""
import os, sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

SRC = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb"
OUT = "/media/user/newMemery/worksapce/aaai_material/model_output/sen12mscr_cloudy_gt_0_3_png/gt"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob(os.path.join(SRC, "*.tif")))
for f in tqdm(files, desc="Converting"):
    with rasterio.open(f) as s:
        img = s.read().astype(np.float32)
    img = np.clip(img, 0, 10000) / 10000.0
    img = np.clip(img * 3.0, 0, 1)
    img = (np.transpose(img, (1,2,0)) * 255).round().astype(np.uint8)
    out_path = os.path.join(OUT, os.path.basename(f).replace(".tif", ".png"))
    Image.fromarray(img).save(out_path)

print(f"Done: {len(files)} → {OUT}/")
