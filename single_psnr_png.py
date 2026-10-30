import numpy as np, torch, torch.nn.functional as F
from torchvision.transforms.functional import resize
from PIL import Image

pred_path = '/media/user/newMemery/worksapce/aaai_material/model_output/ECRformer/pred_rgb/07387_ROIs2017_winter_s2_108_p311.png'
gt_path   = '/home/user/Workspace/3mti/demo/gt.png'

pred = np.array(Image.open(pred_path).convert('RGB'), dtype=np.float32)  # [H, W, C], [0,255]
gt   = np.array(Image.open(gt_path).convert('RGB'), dtype=np.float32)

pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) / 255.0   # [1, C, H, W], [0,1]
gt_t   = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) / 255.0
pred_t = resize(pred_t, [256, 256])
gt_t   = resize(gt_t, [256, 256])

mse = F.mse_loss(pred_t, gt_t).item()
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
print(f'PSNR: {psnr:.4f} dB')
print(f'MSE:  {mse:.8f}')