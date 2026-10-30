#!/usr/bin/env python3
"""消融实验 demo：cloudy | VH | wo joint | wo skip | full，key 对齐"""

import os, sys, re
from PIL import Image
from glob import glob
from tqdm import tqdm

# ========== 路径 ==========
CLOUDY_DIR   = "/media/user/newMemery/worksapce/aaai_material/model_output/sen12mscr_cloudy_bright"
VH_DIR       = "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_20dn/test/vh"

ABLATION_DIRS = [
    ("wo_joint", "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/wo_skip_0_3_png/"),
    ("wo_skip",  "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/wo_joint_attn_0_3_png/"),
    ("full",     "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/complete_output_0_3_png/"),
]

OUT_DIR = "/media/user/newMemery/worksapce/aaai_material/model_output/SarClear/abaltaion_demo/"

PATCHES = [
    ("spring","31","148"), ("spring","31","176"), ("spring","31","528"),
    ("spring","31","611"), ("spring","31","640"),
    ("spring","44","291"), ("spring","44","592"), ("spring","44","622"),
    ("spring","44","697"), ("spring","44","698"), ("spring","44","714"),
    ("spring","44","731"), ("spring","44","732"), ("spring","44","749"),
    ("spring","44","762"), ("spring","44","767"), ("spring","44","797"),
    ("spring","44","798"), ("spring","44","822"),
    ("spring","106","35"), ("spring","106","36"), ("spring","106","49"),
    ("spring","106","50"), ("spring","106","98"), ("spring","106","112"),
    ("spring","106","365"), ("spring","106","420"), ("spring","106","474"),
    ("spring","106","765"), ("spring","106","766"), ("spring","106","779"),
    ("spring","106","794"), ("spring","106","795"), ("spring","106","805"),
    ("spring","106","823"),
    ("spring","123","576"),
    ("summer","73","31"), ("summer","73","35"), ("summer","73","36"),
    ("summer","73","42"), ("summer","73","60"), ("summer","73","61"),
    ("summer","73","97"), ("summer","73","117"), ("summer","73","118"),
    ("summer","73","119"), ("summer","73","126"), ("summer","73","135"),
    ("summer","73","146"), ("summer","73","147"), ("summer","73","163"),
    ("summer","73","176"), ("summer","73","782"),
    ("summer","119","208"), ("summer","119","499"), ("summer","119","501"),
    ("summer","119","527"), ("summer","119","530"), ("summer","119","786"),
    ("summer","119","801"), ("summer","119","802"), ("summer","119","803"),
    ("fall","139","488"), ("fall","139","534"), ("fall","139","535"),
    ("fall","139","541"), ("fall","139","545"), ("fall","139","547"),
    ("fall","139","563"), ("fall","139","564"), ("fall","139","574"),
    ("fall","139","575"), ("fall","139","576"), ("fall","139","593"),
    ("fall","139","600"), ("fall","139","601"), ("fall","139","602"),
    ("fall","139","605"), ("fall","139","656"), ("fall","139","660"),
    ("fall","139","687"),
    ("winter","63","38"), ("winter","63","229"), ("winter","63","466"),
    ("winter","63","612"), ("winter","63","613"), ("winter","63","614"),
    ("winter","63","642"), ("winter","63","672"), ("winter","63","822"),
    ("winter","108","72"), ("winter","108","73"), ("winter","108","75"),
    ("winter","108","76"), ("winter","108","77"), ("winter","108","78"),
    ("winter","108","100"), ("winter","108","101"), ("winter","108","102"),
    ("winter","108","103"), ("winter","108","104"), ("winter","108","105"),
    ("winter","108","106"), ("winter","108","107"), ("winter","108","134"),
    ("winter","108","135"), ("winter","108","136"), ("winter","108","137"),
    ("winter","108","161"), ("winter","108","164"), ("winter","108","190"),
    ("winter","108","191"), ("winter","108","195"), ("winter","108","202"),
    ("winter","108","219"), ("winter","108","220"), ("winter","108","248"),
    ("winter","108","249"), ("winter","108","250"), ("winter","108","251"),
    ("winter","108","252"), ("winter","108","298"), ("winter","108","305"),
    ("winter","108","311"), ("winter","108","334"), ("winter","108","342"),
    ("winter","108","370"), ("winter","108","371"), ("winter","108","392"),
    ("winter","108","426"), ("winter","108","451"), ("winter","108","452"),
    ("winter","108","453"), ("winter","108","481"), ("winter","108","514"),
    ("winter","108","805"),
]


def extract_key(fname):
    name = os.path.basename(fname)
    name = re.sub(r'\.(png|tif|tiff)$', '', name, flags=re.I)
    name = re.sub(r'_(cloudy|pred|rgb|target|SAR|Clear|clear|Cloudy)', '', name, flags=re.I)
    m = re.match(r'ROIs\d+_(\w+)_s2_(\d+)_p(\d+)$', name)
    if m: return (m.group(1), int(m.group(2)), int(m.group(3)))
    m = re.match(r'(\w+)_(\d+)_p(\d+)', name)
    if m: return (m.group(1), int(m.group(2)), int(m.group(3)))
    m = re.match(r'(\w+)_(\d+)_(\d+)', name)
    if m: return (m.group(1), int(m.group(2)), int(m.group(3)))
    return None


def build_dict(folder):
    result = {}
    for f in sorted(glob(os.path.join(folder, "*.png"))):
        k = extract_key(f)
        if k:
            result[k] = f
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cloudy_dict = build_dict(CLOUDY_DIR)
    vh_dict     = build_dict(VH_DIR)
    abl_dicts   = {name: build_dict(d) for name, d in ABLATION_DIRS}

    all_dicts = [cloudy_dict, vh_dict] + [abl_dicts[n] for n,_ in ABLATION_DIRS]

    for key in tqdm(PATCHES, desc="Generating"):
        season, scene, patch = key
        k = (season, int(scene), int(patch))

        columns = []
        for d in all_dicts:
            path = d.get(k)
            if path is None:
                break
            columns.append(Image.open(path).convert("RGB"))

        if len(columns) != len(all_dicts):
            continue

        W, H = columns[0].size
        for j in range(len(columns)):
            if columns[j].size != (W, H):
                columns[j] = columns[j].resize((W, H), Image.LANCZOS)

        combined = Image.new("RGB", (W * len(columns), H))
        for j, col in enumerate(columns):
            combined.paste(col, (j * W, 0))

        out_name = os.path.basename(cloudy_dict[k])
        combined.save(os.path.join(OUT_DIR, out_name))

    print(f"Done → {OUT_DIR}/")


if __name__ == "__main__":
    main()
