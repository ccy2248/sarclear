#!/usr/bin/env python3
"""原地重命名: {season}_scene{scene}_patch{patch}.png → ROIs{ROI}_{season}_s2_{scene}_p{patch}.png"""

import os
import re
from glob import glob

# (season, scene) → ROIs 前缀
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

# 需要重命名的目录列表
DIRS = [
    "/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_20dn/test/vh",
]


def main():
    for folder in DIRS:
        if not os.path.isdir(folder):
            print(f"[WARN] Not found: {folder}")
            continue

        files = sorted(glob(os.path.join(folder, "*_scene*_patch*.*")))
        renamed = 0
        for old_path in files:
            fname = os.path.basename(old_path)
            m = re.match(r"(\w+)_scene(\d+)_patch(\d+)\.(.+)", fname)
            if not m:
                continue
            season, scene, patch, ext = m.group(1), m.group(2), m.group(3), m.group(4)
            roi = SEASON_SCENE_TO_ROI.get((season, scene))
            if roi is None:
                print(f"[WARN] Unknown mapping: {fname}")
                continue

            new_name = f"{roi}_{season}_s2_{scene}_p{patch}.{ext}"
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            renamed += 1

        print(f"{folder}: {renamed} files renamed")
    print("Done.")


if __name__ == "__main__":
    main()
