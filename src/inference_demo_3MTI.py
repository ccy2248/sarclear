#!/usr/bin/env python3
"""Demo 推理脚本：取前 10 张图，输出 PNG / float32 TIF / uint16 TIF 三种格式对比"""

import os
import sys
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from einops import rearrange

# ====== TIF 支持 ======
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[ERROR] rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

# 复用 inference_3MTI 的预处理函数（直接 import）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("SD_TURBO_PATH", "/home/user/Workspace/3mti/models/sd-turbo")
from model import Difix, load_ckpt_from_state_dict_noopt


# ── TIF 读写工具 ────────────────────────────────────────────

def read_tif(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def process_MS(img):
    """Sentinel-2: clip [0,10000] → [0,1]"""
    return np.clip(img, 0.0, 10000.0) / 10000.0


def process_SAR(img, sar_type='vh'):
    """Sentinel-1: clip dB → [0,1]"""
    dB_min, dB_max = (-32.5, 0.0) if sar_type == 'vh' else (-25.0, 0.0)
    img = np.clip(img, dB_min, dB_max)
    return (img - dB_min) / (dB_max - dB_min)


def save_tif_with_profile(path, img_np, ref_tif_path, dtype='float32'):
    """保存 TIF，复制源文件的 CRS/transform 等空间参考"""
    if dtype == 'uint16':
        img_np = np.clip(np.round(img_np), 0, 65535).astype(np.uint16)
    else:
        img_np = img_np.astype(np.float32)

    # 统一为 [C, H, W]
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, :, :]
    elif img_np.ndim == 3 and img_np.shape[-1] in (1, 3):
        img_np = np.transpose(img_np, (2, 0, 1))

    with rasterio.open(ref_tif_path) as src:
        profile = src.profile.copy()

    profile.update(driver='GTiff', height=img_np.shape[1],
                   width=img_np.shape[2], count=img_np.shape[0],
                   dtype=img_np.dtype)

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(img_np)


# ── 加载图像 ────────────────────────────────────────────────

def load_images(path, limit=None):
    exts = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, ext)))
    files = sorted(files)
    return files[:limit] if limit else files


# ── 预处理（同训练管线）──────────────────────────────────────

def preprocess(input_path, ref_path, height, width, device, sar_type='vh'):
    """TIF 模式预处理（与 inference_3MTI 一致）"""
    # cloudy
    cloudy_np = read_tif(input_path)
    cloudy_np = process_MS(cloudy_np)
    x_in = torch.from_numpy(cloudy_np)
    x_in = TF.resize(x_in, [height, width])
    x_in = TF.normalize(x_in, mean=[0.5], std=[0.5])

    # SAR ref
    if ref_path is not None:
        sar_np = read_tif(ref_path)
        sar_np = process_SAR(sar_np, sar_type)
        if sar_np.ndim == 2:
            sar_np = sar_np[np.newaxis, :]
        if sar_np.shape[0] == 1:
            sar_np = np.repeat(sar_np, 3, axis=0)
        x_ref = torch.from_numpy(sar_np)
        x_ref = TF.resize(x_ref, [height, width])
        x_ref = TF.normalize(x_ref, mean=[0.5], std=[0.5])
        cond = torch.stack([x_in, x_ref], dim=0)
    else:
        cond = x_in.unsqueeze(0)

    return cond.unsqueeze(0).to(device)


# ── 主程序 ──────────────────────────────────────────────────

if __name__ == "__main__":
    # ========== 配置区 ==========
    INPUT_DIR  = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb_cloudy"
    REF_DIR    = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/vh"
    TARGET_DIR = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb"
    MODEL_PATH = "/media/user/newMemery/worksapce/crsd/model/3mti/trained_model/checkpoint/sen12mscr_tif_vh_rf.pkl"
    OUT_BASE   = "/home/user/Workspace/3mti/png_float32_uint16_demo"
    N_DEMO     = 10
    PROMPT     = "remove cloud"
    SAR_TYPE   = "vh"
    H, W       = 512, 512
    TIMESTEP   = 199
    # ============================

    # 输出子目录
    for sub in ["png", "float32", "uint16"]:
        os.makedirs(os.path.join(OUT_BASE, sub), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Difix(pretrained_name=None, pretrained_path=None,
                  timestep=TIMESTEP, mv_unet=True)
    model = load_ckpt_from_state_dict_noopt(model, MODEL_PATH)
    model.set_eval()
    print(f"Model loaded: {MODEL_PATH}")

    # 取前 N 张
    cloudy_list = load_images(INPUT_DIR, limit=N_DEMO)
    ref_list    = load_images(REF_DIR, limit=N_DEMO)
    target_list = load_images(TARGET_DIR, limit=N_DEMO)
    assert len(cloudy_list) == len(ref_list) == len(target_list) == N_DEMO, \
        f"Image count mismatch: cloudy={len(cloudy_list)}, ref={len(ref_list)}, target={len(target_list)}"

    print(f"Demo mode: {N_DEMO} images → {OUT_BASE}")

    for in_path, rf_path, gt_path in tqdm(list(zip(cloudy_list, ref_list, target_list)),
                                          desc="Demo inference", total=N_DEMO):
        fname = os.path.basename(in_path)

        # 预处理
        x = preprocess(in_path, rf_path, H, W, device, SAR_TYPE)

        # 推理
        with torch.no_grad():
            y = model(x, prompt=PROMPT)                     # [B=1, V=2, C=3, H, W]

        # ── 真值预处理 (单视图，无 ref) ──
        gt_np = read_tif(gt_path)
        gt_np = process_MS(gt_np)                           # [C, H, W], [0,1]
        gt_tensor = torch.from_numpy(gt_np)
        gt_tensor = TF.resize(gt_tensor, [H, W])
        gt_tensor = TF.normalize(gt_tensor, mean=[0.5], std=[0.5])  # [-1,1]

        # ── 与训练 debug 完全一致的 PNG: 三列 [源图 | 预测 | 真值] ──
        src_vis = rearrange(x, "b v c h w -> b c (v h) w")[0].detach().cpu()   # [3, 1024, 512]
        src_vis = src_vis * 0.5 + 0.5

        pred_vis = rearrange(y, "b v c h w -> b c (v h) w")[0].detach().cpu()   # [3, 1024, 512]
        pred_vis = pred_vis * 0.5 + 0.5

        tgt_vis = gt_tensor * 0.5 + 0.5                                         # [3, 512, 512]
        tgt_vis = tgt_vis.unsqueeze(1).repeat(1, 2, 1, 1)                       # [3, 2, 512, 512] → 重复为 2 视图
        tgt_vis = rearrange(tgt_vis, "c v h w -> c (v h) w")                    # [3, 1024, 512]

        combined = torch.cat([src_vis, pred_vis, tgt_vis], dim=2)               # [3, 1024, 1536]
        output_pil = transforms.ToPILImage()(combined)
        save_base = fname.replace('_cloudy', '_pred').rsplit('.', 1)[0]
        output_pil.save(os.path.join(OUT_BASE, "png", f"{save_base}.png"))

        # ── 预测视图 0 的 [0,1] 数组 (用于 TIF) ──
        out_01 = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()  # [3, 512, 512], [0,1]

        # ── float32 TIF ──
        out_10000 = out_01 * 10000.0
        save_tif_with_profile(
            os.path.join(OUT_BASE, "float32", f"{save_base}.tif"),
            out_10000, in_path, dtype='float32')

        # ── uint16 TIF ──
        save_tif_with_profile(
            os.path.join(OUT_BASE, "uint16", f"{save_base}.tif"),
            out_10000, in_path, dtype='uint16')

    print(f"\nDone! Outputs in {OUT_BASE}/{{png, float32, uint16}}/")
