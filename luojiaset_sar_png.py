#!/usr/bin/env python3
"""LuojiaSET SAR VH → PNG: dB[-32.5,0] 归一化"""
import os, sys, argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)


def convert_level(src_dir, out_dir):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in glob(os.path.join(src_dir, "*.tif"))
                    if not f.endswith(".aux.xml")])
    for f in tqdm(files, desc=os.path.basename(out_dir)):
        with rasterio.open(f) as s:
            img = s.read().astype(np.float32)       # [2, H, W]
        vh = img[1]                                   # VH 波段
        vh = np.clip(vh, -32.5, 0.0)                  # dB 归一化
        vh = (vh + 32.5) / 32.5                       # [0, 1]
        vh = (vh * 255).round().astype(np.uint8)
        # 单通道 → 三通道灰度 PNG (与 RGB 格式一致)
        vh_rgb = np.stack([vh, vh, vh], axis=-1)
        out_name = os.path.basename(f).replace("_SAR","").replace("_sar","") \
                                       .replace(".tif",".png")
        out_path = os.path.join(out_dir, out_name)
        Image.fromarray(vh_rgb).save(out_path)
    print(f"  {os.path.basename(out_dir)}: {len(files)} saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR")
    parser.add_argument("--out_root",  default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_gt_cloudy_0_3_png/sar_vh")
    parser.add_argument("--levels", nargs="+",
                        default=["0-20%","20%-40%","40%-60%","60%-80%","80%-100%"])
    args = parser.parse_args()

    for level in args.levels:
        convert_level(
            os.path.join(args.data_root, level, "s1"),
            os.path.join(args.out_root, level))

    print("All done.")


if __name__ == "__main__":
    main()
