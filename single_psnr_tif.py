import rasterio, numpy as np, torch, torch.nn.functional as F
from torchvision.transforms.functional import resize

pred_path = '/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/model_output/sen12mscr_output/tif_vh_rf_complete_output/winter_108_311_pred.tif'
gt_path  = '/media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_rgb_vh_vv_tif/test/rgb/winter_108_311_rgb.tif'

with rasterio.open(pred_path) as s:
    pred = s.read().astype(np.float32)
    pred = np.clip(pred, 0, 10000) / 10000.0        # [0,10000] → [0,1]
with rasterio.open(gt_path) as s:
    gt = s.read().astype(np.float32)
    gt = np.clip(gt, 0, 10000) / 10000.0

pred_t = torch.from_numpy(pred).unsqueeze(0)         # [1, C, H, W]
gt_t   = torch.from_numpy(gt).unsqueeze(0)
pred_t = resize(pred_t, [256, 256])                  # 预测 512→256
gt_t   = resize(gt_t, [256, 256])

mse = F.mse_loss(pred_t, gt_t).item()
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
print(f'PSNR: {psnr:.4f} dB')