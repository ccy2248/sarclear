#!/usr/bin/env python3
"""LuojiaSET-OSFCR: cloudy + gt → RGB ×3 → PNG (分别输出)"""
import os, sys, re, argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
try:
    import rasterio
except ImportError:
    print("[ERROR] pip install rasterio"); sys.exit(1)

RGB = [3, 2, 1]  # B4/B3/B2


def clean_name(fname):
    """ROIs_01_s2_cloudy_p0576.tif → ROIs_01_s2_p0576"""
    name = os.path.basename(fname)
    name = re.sub(r'\.(tif|tiff)$', '', name, flags=re.I)
    name = re.sub(r'_(cloudy|Clear|clear|Cloudy)', '', name)
    return name + ".png"


def convert_dir(src_dir, out_dir, vmin=0, vmax=10000, bright=3.0):
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in glob(os.path.join(src_dir, "*.tif"))
                    if not f.endswith(".aux.xml")])
    for f in tqdm(files, desc=os.path.basename(out_dir)):
        with rasterio.open(f) as s:
            img = s.read().astype(np.float32)
        rgb = img[RGB]
        rgb = np.clip(rgb, vmin, vmax) / (vmax - vmin)
        rgb = np.clip(rgb * bright, 0, 1)
        rgb = (np.transpose(rgb, (1,2,0)) * 255).round().astype(np.uint8)
        out_path = os.path.join(out_dir, clean_name(f))
        Image.fromarray(rgb).save(out_path)
    print(f"  {os.path.basename(out_dir)}: {len(files)} saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR")
    parser.add_argument("--out_root",  default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_gt_cloudy_0_3_png")
    parser.add_argument("--levels", nargs="+",
                        default=["0-20%","20%-40%","40%-60%","60%-80%","80%-100%"])
    parser.add_argument("--bright_cloudy", type=float, default=2.0, help="cloudy 亮度倍数")
    parser.add_argument("--bright_gt",     type=float, default=3.0, help="gt 亮度倍数")
    args = parser.parse_args()

    for level in args.levels:
        # cloudy
        convert_dir(
            os.path.join(args.data_root, level, "s2_cloudy"),
            os.path.join(args.out_root, "cloudy", level),
            bright=args.bright_cloudy)
        # gt (clear)
        convert_dir(
            os.path.join(args.data_root, level, "s2"),
            os.path.join(args.out_root, "gt", level),
            bright=args.bright_gt)

    print("All done.")


if __name__ == "__main__":
    main()
