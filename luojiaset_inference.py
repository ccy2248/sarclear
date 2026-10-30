#!/usr/bin/env python3
"""LuojiaSET-OSFCR 专用推理脚本：13波段输入 → 3波段模型推理 → 13波段输出"""

import os
import sys
import re
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF

try:
    import rasterio
except ImportError:
    print("[ERROR] rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

os.environ.setdefault("SD_TURBO_PATH", "/home/user/Workspace/3mti/models/sd-turbo")
sys.path.insert(0, "/home/user/Workspace/3mti/src")
from model import Difix, load_ckpt_from_state_dict_noopt

# Sentinel-2 13波段 RGB 索引 (0-based): B4=Red, B3=Green, B2=Blue
RGB_BANDS = [3, 2, 1]


def load_s2_tifs(folder):
    """加载文件夹下所有 .tif（排除 aux.xml）"""
    files = sorted(glob(os.path.join(folder, "*.tif")))
    return [f for f in files if not f.endswith(".aux.xml")]


def process_ms_band(band, vmin=0, vmax=10000):
    """单波段 clip → [0,1]"""
    return np.clip(band, vmin, vmax).astype(np.float32) / (vmax - vmin)


def process_sar_band(band):
    """SAR VH dB → clip [-32.5, 0] → [0,1]"""
    return np.clip(band, -32.5, 0.0).astype(np.float32) / 32.5 + 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/media/user/newMemery/worksapce/crsd/data/LuojiaSET-OSFCR",
                        help="LuojiaSET-OSFCR 根目录")
    parser.add_argument("--model_path", type=str,
                        default="/media/user/newMemery/worksapce/crsd/model/3mti/trained_model/checkpoint/sen12mscr_tif_vh_rf.pkl")
    parser.add_argument("--output_root", type=str,
                        default="/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/tif")
    parser.add_argument("--prompt", type=str, default="remove cloud")
    parser.add_argument("--cloud_folders", nargs="+",
                        default=["0-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"],
                        help="要处理的云覆盖率子目录")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--timestep", type=int, default=199)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Difix(pretrained_name=None, pretrained_path=None,
                  timestep=args.timestep, mv_unet=True)
    model = load_ckpt_from_state_dict_noopt(model, args.model_path)
    model.set_eval()
    print(f"Model loaded: {args.model_path}")

    for cloud_folder in args.cloud_folders:
        cloudy_dir = os.path.join(args.data_root, cloud_folder, "s2_cloudy")
        sar_dir    = os.path.join(args.data_root, cloud_folder, "s1")
        out_dir    = os.path.join(args.output_root, cloud_folder, "pred")

        if not os.path.isdir(cloudy_dir):
            print(f"[SKIP] {cloud_folder}: no data")
            continue

        os.makedirs(out_dir, exist_ok=True)

        cloudy_files = load_s2_tifs(cloudy_dir)
        sar_files    = load_s2_tifs(sar_dir)

        # 按 ROI + patch ID 对齐
        def key(f):
            """ROIs_01_s2_cloudy_p0576.tif 或 ROIs_01_s1_SAR_p0576.tif → ROIs_01_p0576"""
            name = os.path.basename(f)
            name = re.sub(r"_(cloudy|Cloudy|Clear|clear|SAR|sar)", "", name, flags=re.I)
            name = re.sub(r"_s[12]_", "_", name)  # 去掉 s1_ / s2_
            name = name.replace(".tif", "").replace(".tiff", "")
            return name
        cloudy_dict = {key(f): f for f in cloudy_files}
        sar_dict    = {key(f): f for f in sar_files}
        common = sorted(set(cloudy_dict) & set(sar_dict))

        print(f"[{cloud_folder}] {len(common)} paired images → {out_dir}/")

        for pid in tqdm(common, desc=f"  {cloud_folder}"):
            c_path = cloudy_dict[pid]
            s_path = sar_dict[pid]

            # ── 读取 ──
            with rasterio.open(c_path) as src:
                cloudy_all = src.read().astype(np.float32)  # [13, H, W]
                profile = src.profile.copy()
            with rasterio.open(s_path) as src:
                sar_all = src.read().astype(np.float32)      # [2, H, W]

            # ── 提取 RGB + SAR VH ──
            cloudy_rgb = cloudy_all[RGB_BANDS]               # [3, H, W]
            for i in range(3):
                cloudy_rgb[i] = process_ms_band(cloudy_rgb[i])

            sar_vh = sar_all[1]                              # VH 波段
            sar_vh = process_sar_band(sar_vh)
            sar_3ch = np.stack([sar_vh] * 3, axis=0)         # [3, H, W]

            # ── 预处理 ──
            x_in = torch.from_numpy(cloudy_rgb)
            x_in = TF.resize(x_in, [args.height, args.width])
            x_in = TF.normalize(x_in, mean=[0.5], std=[0.5])

            x_ref = torch.from_numpy(sar_3ch)
            x_ref = TF.resize(x_ref, [args.height, args.width])
            x_ref = TF.normalize(x_ref, mean=[0.5], std=[0.5])

            cond = torch.stack([x_in, x_ref], dim=0).unsqueeze(0).to(device)  # [1, 2, 3, H, W]

            # ── 推理 ──
            with torch.no_grad():
                y = model(cond, prompt=args.prompt)           # [1, 2, 3, H, W]

            # ── 后处理：RGB 输出 → [0,1] → [0,10000] ──
            out_rgb = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()  # [3, 512, 512]

            # Resize 回原尺寸
            h, w = cloudy_rgb.shape[1], cloudy_rgb.shape[2]
            out_resized = np.stack([
                TF.resize(torch.from_numpy(out_rgb[i:i+1]), [h, w]).numpy()
                for i in range(3)
            ]).squeeze(1)  # [3, H, W]

            out_scaled = out_resized * 10000.0  # [0, 10000]

            # ── 保存 3 波段 RGB TIF ──
            out_name = os.path.basename(c_path)
            out_name = out_name.replace("_cloudy", "").replace("_Cloudy", "")
            out_name = out_name.replace("_Clear", "").replace("_clear", "")
            out_name = out_name.replace("_SAR", "").replace("_sar", "")
            out_path = os.path.join(out_dir, out_name)

            profile.update(driver="GTiff", count=3, dtype="float32",
                           compress="lzw", height=h, width=w)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_scaled.astype(np.float32))

        print(f"  [{cloud_folder}] done → {out_dir}/")

    print("\nAll done!")


if __name__ == "__main__":
    main()
