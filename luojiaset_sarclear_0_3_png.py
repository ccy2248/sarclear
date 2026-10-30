#!/usr/bin/env python3
"""SarClear LuojiaSET 预测 TIF → PNG: [0,10000] → ×3 → PNG"""
import os, sys, argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)


def convert_level(src_dir, out_dir, bright=3.0, vmin=0, vmax=10000):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob(os.path.join(src_dir, "*.tif")))
    for f in tqdm(files, desc=os.path.basename(out_dir)):
        with rasterio.open(f) as s:
            img = s.read().astype(np.float32)
        img = np.clip(img, vmin, vmax) / (vmax - vmin)
        img = np.clip(img * bright, 0, 1)
        img = (np.transpose(img, (1,2,0)) * 255).round().astype(np.uint8)
        out_path = os.path.join(out_dir, os.path.basename(f).replace(".tif", ".png"))
        Image.fromarray(img).save(out_path)
    print(f"  {os.path.basename(out_dir)}: {len(files)} saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/tif")
    parser.add_argument("--out_root", default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/output_0_3_png")
    parser.add_argument("--levels", nargs="+",
                        default=["0-20%","20%-40%","40%-60%","60%-80%","80%-100%"])
    parser.add_argument("--bright", type=float, default=3.0)
    args = parser.parse_args()

    for level in args.levels:
        convert_level(
            os.path.join(args.src_root, level, "pred"),
            os.path.join(args.out_root, level),
            bright=args.bright)

    print("All done.")


if __name__ == "__main__":
    main()
