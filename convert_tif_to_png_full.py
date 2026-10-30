#!/usr/bin/env python3
"""将 TIF 预测结果 0~10000 线性映射 → PNG，并按参考命名规则重命名"""

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

try:
    import rasterio
except ImportError:
    print("[ERROR] rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ========== 配置 ==========
SRC_DIR = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/model_output/sen12mscr_output/tif_vh_rf_complete_output"
OUT_DIR = "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/full_rgb_png"

# (season, scene) → ROIs 前缀（从参考目录提取）
SEASON_SCENE_TO_ROI = {
    ("spring", "31"):  "ROIs1158",
    ("spring", "44"):  "ROIs1158",
    ("spring", "106"): "ROIs1158",
    ("spring", "123"): "ROIs1158",
    ("spring", "140"): "ROIs1158",
    ("summer", "73"):  "ROIs1868",
    ("summer", "119"): "ROIs1868",
    ("fall",   "139"): "ROIs1970",
    ("winter", "63"):  "ROIs2017",
    ("winter", "108"): "ROIs2017",
}
# ==========================


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)  # [C, H, W]


def parse_filename(fname):
    """fall_139_100_pred.tif → ('fall', '139', '100')"""
    stem = fname.replace(".tif", "").replace(".tiff", "").replace("_pred", "")
    parts = stem.split("_")
    # Expected: season_scene_patch, e.g. fall_139_100
    if len(parts) >= 3:
        season = parts[0]
        scene  = parts[1]
        patch  = parts[2]
        return season, scene, patch
    return None, None, None


def build_target_name(season, scene, patch):
    """spring, 106, 100 → ROIs1158_spring_s2_106_p100.png"""
    roi = SEASON_SCENE_TO_ROI.get((season, scene))
    if roi is None:
        print(f"[WARN] Unknown (season={season}, scene={scene}), skipping")
        return None
    return f"{roi}_{season}_s2_{scene}_p{patch}.png"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tif_files = sorted(glob(os.path.join(SRC_DIR, "*.tif")) + glob(os.path.join(SRC_DIR, "*.tiff")))

    skipped = 0
    for path in tqdm(tif_files, desc="Converting"):
        fname = os.path.basename(path)
        season, scene, patch = parse_filename(fname)

        target_name = build_target_name(season, scene, patch)
        if target_name is None:
            skipped += 1
            continue

        # TIF [0,10000] → clip → /10000 → [0,1] → ×255 → uint8 PNG
        img_np = read_tif(path)
        img_np = np.clip(img_np, 0, 10000) / 10000.0     # [0,1]
        img_np = np.transpose(img_np, (1, 2, 0))          # [H, W, C]
        img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(OUT_DIR, target_name))

    print(f"Done: {len(tif_files) - skipped} saved, {skipped} skipped → {OUT_DIR}/")


if __name__ == "__main__":
    main()
