#!/usr/bin/env python3
"""Demo 可视化脚本：三组预测 × 10 张，输出 [cloudy | pred | gt] + [VH] 对比图"""

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import cv2

try:
    import rasterio
except ImportError:
    print("[ERROR] rasterio not installed. Run: pip install rasterio")
    sys.exit(1)


# ========== 配置 ==========
CLOUDY_DIR  = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb_cloudy"
VH_DIR      = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/vh"
GT_DIR      = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb"

PRED_DIRS = {
    "l2x100":      "/home/user/Workspace/3mti/sen12mscr_output/tif_vh_l2x100_output",
    "rf_complete": "/home/user/Workspace/3mti/sen12mscr_output/tif_vh_rf_complete_output",
    "cfg0_8":      "/home/user/Workspace/3mti/sen12mscr_output/tif_vh_cfg_output/cfg0_8",
}

OUT_BASE = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/model_output/demo"
N_DEMO   = None    # None = 全部
H, W     = 256, 256
# ==========================


def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)  # [C, H, W]


def resize_to(img, target_h, target_w):
    """resize [C, H, W] or [H, W] → [C, target_h, target_w] using cv2 bilinear"""
    orig_ndim = img.ndim
    if img.ndim == 2:
        img = img[np.newaxis, :, :]  # [1, H, W]
    # cv2.resize expects [H, W, C], so transpose
    img_t = np.transpose(img, (1, 2, 0))  # [H, W, C]
    img_t = cv2.resize(img_t, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if img_t.ndim == 2:
        img_t = img_t[:, :, np.newaxis]
    img = np.transpose(img_t, (2, 0, 1))  # [C, target_h, target_w]
    if orig_ndim == 2:
        img = img[0]  # back to [target_h, target_w]
    return img


def load_sorted_tifs(folder, limit=None):
    files = sorted(glob(os.path.join(folder, "*.tif")) + glob(os.path.join(folder, "*.tiff")))
    return files[:limit] if limit else files


def percentile_norm(img, lo=2, hi=98):
    """按通道 2/98 百分位 min-max 归一化 → [0,1]"""
    img = img.copy()
    for c in range(img.shape[0]):
        band = img[c]
        vmin, vmax = np.percentile(band, lo), np.percentile(band, hi)
        if vmax > vmin:
            img[c] = np.clip((band - vmin) / (vmax - vmin), 0, 1)
        else:
            img[c] = 0
    return img


def sar_to_png(sar_np):
    """SAR VH dB → [-32.5, 0] clip → [0,1] → uint8 [H, W, 3]"""
    sar_np = np.clip(sar_np, -32.5, 0.0)
    sar_np = (sar_np + 32.5) / 32.5  # [0, 1]
    if sar_np.ndim == 3:
        sar_np = sar_np[0]  # 取第一通道
    sar_uint8 = (sar_np * 255).astype(np.uint8)
    return np.stack([sar_uint8] * 3, axis=-1)  # [H, W, 3] 灰度转 RGB


def rgb_tensor_to_png(rgb_np):
    """RGB [C, H, W] float → 2/98% norm → uint8 [H, W, 3]"""
    rgb_np = percentile_norm(rgb_np, lo=2, hi=98)
    rgb_np = np.transpose(rgb_np, (1, 2, 0))  # [H, W, C]
    return (rgb_np * 255).clip(0, 255).astype(np.uint8)


def make_demo_panel(cloudy_np, pred_np, gt_np, vh_np):
    """构建对比图: 上行 cloudy|pred|gt, 下行 VH|VH|VH"""
    # 上排 RGB
    row_top = np.concatenate([
        rgb_tensor_to_png(cloudy_np),
        rgb_tensor_to_png(pred_np),
        rgb_tensor_to_png(gt_np),
    ], axis=1)  # [H, W*3, 3]

    # 下排 VH
    vh_vis = sar_to_png(vh_np)
    row_bot = np.concatenate([vh_vis, vh_vis, vh_vis], axis=1)  # [H, W*3, 3]

    # 添加标签行（简单用白色分隔线，用户可在 caption 中查看）
    spacer = np.ones((4, W * 3, 3), dtype=np.uint8) * 255

    combined = np.concatenate([row_top, spacer, row_bot], axis=0)
    return Image.fromarray(combined)


def main():
    for model_name, pred_dir in PRED_DIRS.items():
        out_subdir = os.path.join(OUT_BASE, model_name)
        os.makedirs(out_subdir, exist_ok=True)
        print(f"\n{'='*60}\n  [{model_name}]  {pred_dir}\n{'='*60}")

        cloudy_list = load_sorted_tifs(CLOUDY_DIR, N_DEMO)
        pred_list   = load_sorted_tifs(pred_dir, N_DEMO)
        gt_list     = load_sorted_tifs(GT_DIR, N_DEMO)
        vh_list     = load_sorted_tifs(VH_DIR, N_DEMO)

        assert len(cloudy_list) == len(pred_list) == len(gt_list) == len(vh_list), \
            f"Count mismatch: cloudy={len(cloudy_list)} pred={len(pred_list)} gt={len(gt_list)} vh={len(vh_list)}"

        for c_path, p_path, g_path, v_path in tqdm(
            list(zip(cloudy_list, pred_list, gt_list, vh_list)),
            desc=f"  {model_name}", total=N_DEMO
        ):
            cloudy_np = resize_to(read_tif(c_path), H, W)
            pred_np   = resize_to(read_tif(p_path), H, W)
            gt_np     = resize_to(read_tif(g_path), H, W)
            vh_np     = resize_to(read_tif(v_path), H, W)

            panel = make_demo_panel(cloudy_np, pred_np, gt_np, vh_np)
            save_name = os.path.basename(p_path).replace("_pred", "").replace(".tif", ".png")
            panel.save(os.path.join(out_subdir, save_name))

        print(f"  → saved {N_DEMO} panels to {out_subdir}/")

    print(f"\nDone! All panels in {OUT_BASE}/{{l2x100, rf_complete, cfg0_8}}/")


if __name__ == "__main__":
    main()
