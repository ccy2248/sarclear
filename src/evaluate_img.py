import pyiqa
import os
import argparse
from pathlib import Path
import torch
import util_image
import tqdm
import torch.nn.functional as F
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import json
import numpy as np

# ====== TIF 支持 ======
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

print(pyiqa.list_models())


def read_image_tensor(path, target_size=(256, 256)):
    """统一读取图像 → [1, C, H, W] float32 tensor, 值域 [0,1]"""
    path_str = str(path)
    if path_str.lower().endswith(('.tif', '.tiff')) and HAS_RASTERIO:
        with rasterio.open(path_str) as src:
            img = src.read().astype(np.float32)          # [C, H, W]
        # Sentinel-2: clip [0, 10000] → [0, 1]
        img = np.clip(img, 0.0, 10000.0) / 10000.0
        img = torch.from_numpy(img).unsqueeze(0)          # [1, C, H, W]
    else:
        # PNG/JPEG: 走原有 cv2 路径
        im = util_image.imread(path_str, chn='rgb', dtype='float32')  # [H, W, C], [0,1]
        img = util_image.img2tensor(im)                                # [1, C, H, W]
    img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
    return img.cuda()


def load_image_paths(folder, exts=("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")):
    """加载文件夹下所有图像路径（支持 TIF）"""
    folder = Path(folder) if not isinstance(folder, Path) else folder
    files = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return sorted(files)


def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    # metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    # metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    # metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    # metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = load_image_paths(ref_path)
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        # metric_paired_dict["dists"]=pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = load_image_paths(in_path)
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    resultall = {
    'in_name': [],
    'ssim': [],
    'psnr': [],
    'lpips': [],
    'mae': [],
    'sam': [],
    # 'musiq': [],
    # 'maniqa': [],
    # 'clipiqa': [],
    # 'dists': [],
    # 'niqe': []
}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None
        
        resultall['in_name'].append(os.path.basename(_in_path))
        im_in_tensor = read_image_tensor(_in_path)
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                res = metric(im_in_tensor).item()
                result[key] = result.get(key, 0) + res
                resultall[key].append(res)
        
        if ref_path is not None:
            im_ref_tensor = read_image_tensor(_ref_path)
            for key, metric in metric_paired_dict.items():
                res = metric(im_in_tensor, im_ref_tensor).item()
                result[key] = result.get(key, 0) + res
                resultall[key].append(res)

            # ---- MAE & SAM (自定义计算, 值域 [0,1]) ----
            mae_val = F.l1_loss(im_in_tensor, im_ref_tensor).item()
            result['mae'] = result.get('mae', 0) + mae_val
            resultall['mae'].append(mae_val)

            eps = 1e-8
            a = im_in_tensor.squeeze(0)   # [C, H, W]
            b = im_ref_tensor.squeeze(0)
            dot = (a * b).sum(dim=0)
            cos_sim = dot / (a.norm(dim=0) * b.norm(dim=0) + eps)
            cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)
            sam_val = torch.acos(cos_sim).mean().item()  # 弧度
            result['sam'] = result.get('sam', 0) + sam_val
            resultall['sam'].append(sam_val)
    
    # if ref_path is not None:
    #     fid_metric = pyiqa.create_metric('fid')
    #     result['fid'] = fid_metric(in_path, ref_path)

    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
        else:
            print(f"{key}: {res/len(lr_path_list):.5f}")
    
    jsonpath = os.path.join(args.output)
    with open(jsonpath, 'w') as f:
        json.dump(resultall, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default='./')
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    
