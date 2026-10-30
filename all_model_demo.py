#!/usr/bin/env python3
"""生成模型对比 demo：cloudy | VH | 6模型 | rgb 横向拼接"""

import os
import sys
import re
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm

# ========== 配置 ==========
CLOUDY_DIR = "/media/user/newMemery/worksapce/aaai_material/model_output/sen12mscr_cloudy_bright"
VH_DIR     = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_20dn/test/vh"
RGB_DIR    = "/media/user/newMemery/worksapce/aaai_material/model_output/sen12mscr_cloudy_gt_0_3_png/gt/"

# 模型输出 (按指定顺序)
MODEL_DIRS = [
    ("Dsen2-CR",   "/media/user/newMemery/worksapce/aaai_material/model_output/Dsen2-CR/rgb_png/"),
    ("Diffcr",     "/media/user/newMemery/worksapce/aaai_material/model_output/Diffcr/rgb_png"),
    ("UnCRtainTS", "/media/user/newMemery/worksapce/aaai_material/model_output/UnCRtainTS/output_0_3_png"),
    ("SarClear",   "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/complete_output_0_3_png/"),
]

OUT_DIR = "/media/user/newMemery/worksapce/aaai_material/model_output/all_model_rgb_demo4"
N_DEMO  = 7899  # 全量
# ==========================


def extract_key(fname):
    """从任意命名格式提取 (season, scene, patch) 作为对齐 key"""
    name = os.path.basename(fname)
    # 去掉后缀和修饰词
    name = re.sub(r'\.(png|tif|tiff)$', '', name, flags=re.I)
    name = re.sub(r'_(cloudy|pred|rgb|target|SAR|Clear|clear|Cloudy)', '', name, flags=re.I)
    # ROIs 格式: ROIs1158_spring_s2_106_p100 → spring, 106, 100
    m = re.match(r'ROIs\d+_(\w+)_s2_(\d+)_p(\d+)$', name)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    # {s}_{c}_p{p}_pred: fall_139_p100_pred → fall, 139, 100
    m = re.match(r'(\w+)_(\d+)_p(\d+)', name)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    # {s}_{c}_{p}: fall_139_100 → fall, 139, 100
    m = re.match(r'(\w+)_(\d+)_(\d+)', name)
    if m:
        return (m.group(1), int(m.group(2)), int(m.group(3)))
    # scene_patch: 139_100 → (unknown, 139, 100)
    m = re.match(r'(\d+)_(\d+)', name)
    if m:
        return ('unknown', int(m.group(1)), int(m.group(2)))
    return None


def load_sorted_by_key(folder):
    """加载目录下所有 PNG，按 (season,scene,patch) 排序，返回 {key: path}"""
    all_files = sorted(glob(os.path.join(folder, "*.png")))
    result = {}
    for f in all_files:
        k = extract_key(f)
        if k:
            result[k] = f
    return result


def add_label(img, text, font_size=16):
    """在图像顶部添加白色文字标签"""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.width - tw) // 2
    # 半透明背景
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x - 4, 2, x + tw + 4, th + 6], fill=(0, 0, 0, 160))
    img.paste(overlay, (0, 0), overlay)
    draw.text((x, 4), text, fill=(255, 255, 255), font=font)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载所有目录，key 为 (season, scene, patch)
    cloudy_dict = load_sorted_by_key(CLOUDY_DIR)
    vh_dict     = load_sorted_by_key(VH_DIR)
    rgb_dict    = load_sorted_by_key(RGB_DIR)
    model_dicts = {name: load_sorted_by_key(d) for name, d in MODEL_DIRS}

    # 取所有目录共同拥有的 key
    common_keys = sorted(set(cloudy_dict) & set(vh_dict) & set(rgb_dict))
    for name, d in model_dicts.items():
        common_keys = sorted(set(common_keys) & set(d))

    common_keys = common_keys[:N_DEMO]
    print(f"Aligned: {len(common_keys)} images (N_DEMO={N_DEMO})")

    for key in tqdm(common_keys, desc="Demo"):
        col_cloudy = Image.open(cloudy_dict[key]).convert("RGB")
        col_vh     = Image.open(vh_dict[key]).convert("RGB")
        col_rgb    = Image.open(rgb_dict[key]).convert("RGB")

        W, H = col_cloudy.size
        col_vh  = col_vh.resize((W, H), Image.LANCZOS)
        col_rgb = col_rgb.resize((W, H), Image.LANCZOS)

        columns = [col_cloudy, col_vh]
        labels  = ["Cloudy", "VH"]

        for name, _ in MODEL_DIRS:
            col = Image.open(model_dicts[name][key]).convert("RGB")
            if col.size != (W, H):
                col = col.resize((W, H), Image.LANCZOS)
            columns.append(col)
            labels.append(name)

        columns.append(col_rgb)
        labels.append("GT")

        for j, (col, lbl) in enumerate(zip(columns, labels)):
            columns[j] = add_label(col, lbl)

        combined = Image.new("RGB", (W * len(columns), H))
        for j, col in enumerate(columns):
            combined.paste(col, (j * W, 0))

        # 输出命名用第一列的文件名
        out_name = os.path.basename(cloudy_dict[key])
        combined.save(os.path.join(OUT_DIR, out_name))

    print(f"Done → {OUT_DIR}/ ({len(common_keys)} panels)")


if __name__ == "__main__":
    main()
