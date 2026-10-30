#!/usr/bin/env python3
"""LuojiaSET demo：cloudy | SAR | SarClear | GT，按 ROI+patch 对齐"""

import os, sys, re
from PIL import Image
from glob import glob
from tqdm import tqdm

# ========== 路径 ==========
CLOUDY_ROOT   = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_gt_cloudy_0_3_png/cloudy"
SAR_ROOT      = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_gt_cloudy_0_3_png/sar_vh"
GT_ROOT       = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_gt_cloudy_0_3_png/gt"
SARCLEAR_ROOT = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/output_0_3_png"

OUT_DIR = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear/lunwen_demo"

LEVELS = ["0-20%","20%-40%","40%-60%","60%-80%","80%-100%"]


def extract_key_luojia(fname):
    """ROIs_01_s2_p0576.png → ROIs_01_p0576 (统一 key)"""
    name = os.path.basename(fname)
    name = re.sub(r'\.(png|tif|tiff)$', '', name, flags=re.I)
    name = re.sub(r'_s[12]_', '_', name)  # 去掉 s1_/s2_
    return name


def build_dict(folder):
    result = {}
    if not os.path.isdir(folder):
        return result
    for f in sorted(glob(os.path.join(folder, "*.png"))):
        k = extract_key_luojia(f)
        if k:
            result[k] = f
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 先收集所有层级的任务
    tasks = []
    for level in LEVELS:
        cloudy_dict   = build_dict(os.path.join(CLOUDY_ROOT, level))
        sar_dict      = build_dict(os.path.join(SAR_ROOT, level))
        gt_dict       = build_dict(os.path.join(GT_ROOT, level))
        sarclear_dict = build_dict(os.path.join(SARCLEAR_ROOT, level))

        all_dicts = [cloudy_dict, sar_dict, sarclear_dict, gt_dict]
        common = sorted(set(cloudy_dict) & set(sar_dict) & set(gt_dict) & set(sarclear_dict))

        level_out = os.path.join(OUT_DIR, level)
        os.makedirs(level_out, exist_ok=True)

        for k in common:
            tasks.append((k, level_out, [d[k] for d in all_dicts]))
        print(f"[{level}] {len(common)} images")

    # 统一进度条处理
    for k, level_out, paths in tqdm(tasks, desc="Generating"):
        columns = [Image.open(p).convert("RGB") for p in paths]
        W, H = columns[0].size
        for j in range(len(columns)):
            if columns[j].size != (W, H):
                columns[j] = columns[j].resize((W, H), Image.LANCZOS)
        combined = Image.new("RGB", (W * len(columns), H))
        for j, col in enumerate(columns):
            combined.paste(col, (j * W, 0))
        combined.save(os.path.join(level_out, f"{k}.png"))

    print(f"\nDone → {OUT_DIR}/")


if __name__ == "__main__":
    main()
