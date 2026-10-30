#!/usr/bin/env python3
"""LuojiaSET-OSFCR: 13波段 S2 → RGB 3波段 → [0,1]线性映射 → PNG"""
import os, sys, argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

RGB = [3, 2, 1]  # Sentinel-2 B4/B3/B2


def process(src_dir, out_dir, vmin=0, vmax=10000):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in glob(os.path.join(src_dir, "*.tif"))
                    if not f.endswith(".aux.xml")])
    for f in tqdm(files, desc=os.path.basename(out_dir)):
        with rasterio.open(f) as s:
            img = s.read().astype(np.float32)          # [13, H, W]
        rgb = img[RGB]                                  # [3, H, W]
        rgb = np.clip(rgb, vmin, vmax) / (vmax - vmin)  # [0,1]
        rgb = (np.transpose(rgb, (1,2,0)) * 255).round().astype(np.uint8)
        out_path = os.path.join(out_dir,
            os.path.basename(f).replace("_cloudy","").replace("_Cloudy","")
                                .replace("_Clear","").replace("_clear","")
                                .replace(".tif",".png"))
        Image.fromarray(rgb).save(out_path)
    print(f"  {os.path.basename(out_dir)}: {len(files)} saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR")
    parser.add_argument("--out_root",  default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/rgb_png")
    parser.add_argument("--levels", nargs="+",
                        default=["0-20%","20%-40%","40%-60%","60%-80%","80%-100%"])
    parser.add_argument("--types", nargs="+", default=["s2", "s2_cloudy"],
                        help="s2=真值, s2_cloudy=有云图")
    parser.add_argument("--vmin", type=float, default=0)
    parser.add_argument("--vmax", type=float, default=10000)
    args = parser.parse_args()

    for level in args.levels:
        for typ in args.types:
            src = os.path.join(args.data_root, level, typ)
            out = os.path.join(args.out_root, level, typ)
            process(src, out, args.vmin, args.vmax)

    print("All done.")


if __name__ == "__main__":
    main()
