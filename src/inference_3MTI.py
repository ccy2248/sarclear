import os
import imageio
import argparse
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from einops import rearrange, repeat

# ====== TIF 卫星图支持 ======
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed, .tif support disabled")


def read_tif(path):
    """rasterio 读取 tif → float32 numpy [C, H, W]"""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def process_MS(img):
    """Sentinel-2 RGB/cloudy: clip [0,10000] → rescale → [0,1]"""
    img = np.clip(img, 0.0, 10000.0)
    img = img / 10000.0
    return img  # [0, 1]


def process_SAR(img, sar_type='vh'):
    """Sentinel-1 SAR VH/VV: clip dB → rescale → [0,1]"""
    if sar_type == 'vh':
        dB_min, dB_max = -32.5, 0.0   # VH 交叉极化
    else:
        dB_min, dB_max = -25.0, 0.0   # VV 同极化
    img = np.clip(img, dB_min, dB_max)
    img = (img - dB_min) / (dB_max - dB_min)
    return img  # [0, 1]


def save_tif(path, img, dtype='uint16'):
    """保存 numpy 数组为 TIF 文件
    Args:
        path: 输出路径
        img: [H, W] 或 [H, W, C], 值域 [0, 10000]
        dtype: 'float32' 无损保存 / 'uint16' 与原始格式一致
    """
    if dtype == 'uint16':
        img = np.clip(np.round(img), 0, 65535).astype(np.uint16)
    else:  # float32
        img = img.astype(np.float32)

    # [H, W] → [1, H, W] or [H, W, C] → [C, H, W]
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    elif img.ndim == 3 and img.shape[-1] in (1, 3):
        img = np.transpose(img, (2, 0, 1))

    with rasterio.open(path, 'w',
                       driver='GTiff',
                       height=img.shape[1],
                       width=img.shape[2],
                       count=img.shape[0],
                       dtype=img.dtype) as dst:
        dst.write(img)


# ====== CFG 潜空间推理 ======

def encode_prompt_cfg(model, prompt, num_views):
    """编码 prompt → [B*V, N, C] text embeddings"""
    caption_tokens = model.tokenizer(
        prompt,
        max_length=model.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(model.text_encoder.device)
    caption_enc = model.text_encoder(caption_tokens)[0]
    return repeat(caption_enc, "b n c -> (b v) n c", v=num_views)


def predict_denoised_latent(model, z, caption_enc):
    """单步 UNet 去噪: z → z_denoised"""
    model_pred = model.unet(z, model.timesteps, encoder_hidden_states=caption_enc).sample
    return model.sched.step(model_pred, model.timesteps, z, return_dict=True).prev_sample


def decode_latent(model, z_denoised, skip_acts, num_views):
    """VAE 解码: z_denoised → [B, V, C, H, W] image, 值域 [-1,1]"""
    model.vae.decoder.incoming_skip_acts = skip_acts
    output_image = model.vae.decode(z_denoised / model.vae.config.scaling_factor).sample
    output_image = output_image.clamp(-1, 1)
    return rearrange(output_image, "(b v) c h w -> b v c h w", v=num_views)


def run_latent_cfg(model, x, pos_prompt, neg_prompt, cfg_scales):
    """CFG 潜空间插值: z_cfg = z_neg + cfg * (z_pos - z_neg)
    Returns: {cfg: tensor[B,V,C,H,W]} dict
    """
    num_views = x.shape[1]
    x_flat = rearrange(x, "b v c h w -> (b v) c h w")

    # 编码一次，正负分支共享同一个 latent sample
    z = model.vae.encode(x_flat).latent_dist.sample() * model.vae.config.scaling_factor
    skip_acts = model.vae.encoder.current_down_blocks

    caption_pos = encode_prompt_cfg(model, pos_prompt, num_views)
    caption_neg = encode_prompt_cfg(model, neg_prompt, num_views)

    z_pos = predict_denoised_latent(model, z, caption_pos)
    z_neg = predict_denoised_latent(model, z, caption_neg)

    outputs = {}
    for cfg in cfg_scales:
        z_cfg = z_neg + cfg * (z_pos - z_neg)
        outputs[cfg] = decode_latent(model, z_cfg, skip_acts, num_views)
    return outputs


os.environ.setdefault("SD_TURBO_PATH", "/home/user/Workspace/3mti/models/sd-turbo")

from model import Difix, load_ckpt_from_state_dict_noopt


def load_images(path):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, ext)))
    files = sorted(files)
    return files

def preprocess_like_training(input_path: str,
                             ref_path: str | None,
                             height: int,
                             width: int,
                             device: torch.device,
                             use_tif: bool = False,
                             sar_type: str = 'vh') -> torch.Tensor:

    if use_tif and HAS_RASTERIO:
        # ---- TIF 模式: 卫星图归一化 ----
        # 待去云图 (Sentinel-2 MS)
        cloudy_np = read_tif(input_path)          # [C, H, W] float32
        cloudy_np = process_MS(cloudy_np)         # → [0, 1]
        x_in = torch.from_numpy(cloudy_np)        # [C, H, W]
        x_in = TF.resize(x_in, [height, width])   # 直接 resize，不做退化模拟
        x_in = TF.normalize(x_in, mean=[0.5], std=[0.5])

        if ref_path is not None:
            # 参考图 (Sentinel-1 SAR)
            sar_np = read_tif(ref_path)               # [1, H, W] or [C, H, W]
            sar_np = process_SAR(sar_np, sar_type)    # → [0, 1]
            # 单通道 → 3 通道
            if sar_np.ndim == 2:
                sar_np = sar_np[np.newaxis, ...]
            if sar_np.shape[0] == 1:
                sar_np = np.repeat(sar_np, 3, axis=0)
            x_ref = torch.from_numpy(sar_np)
        else:
            x_ref = None
    else:
        # ---- PNG/JPG 模式 ----
        inp = Image.open(input_path).convert("RGB")
        x_in = TF.to_tensor(inp)
        x_in = TF.resize(x_in, [height, width])    # 直接 resize 到目标尺寸
        x_in = TF.normalize(x_in, mean=[0.5], std=[0.5])

        if ref_path is not None:
            ref = Image.open(ref_path).convert("RGB")
            x_ref = TF.to_tensor(ref)
        else:
            x_ref = None

    if x_ref is not None:
        x_ref = TF.resize(x_ref, [height, width])
        x_ref = TF.normalize(x_ref, mean=[0.5], std=[0.5])
        cond = torch.stack([x_in, x_ref], dim=0)
    else:
        cond = x_in.unsqueeze(0)

    cond = cond.unsqueeze(0).to(device)
    return cond


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--ref_image', type=str, default=None, help='Path to reference image or directory')
    parser.add_argument('--height', type=int, default=512, help='Network input height')
    parser.add_argument('--width', type=int, default=512, help='Network input width')
    parser.add_argument('--prompt', type=str, default="remove degradation", help='Base prompt')
    parser.add_argument('--model_name', type=str, default=None, help='HF model name (optional)')
    parser.add_argument('--model_path', type=str, default=None, help='Checkpoint path (.pkl)')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--timestep', type=int, default=199, help='Diffusion timestep')
    parser.add_argument('--video', action='store_true', help='Save results as video')
    parser.add_argument("--mv_unet", action="store_true")
    parser.add_argument('--use_tif', action='store_true', help='Use TIF satellite image preprocessing')
    parser.add_argument('--sar_type', type=str, default='vh', choices=['vh', 'vv'], help='SAR polarization type (vh/vv)')
    parser.add_argument('--save_dtype', type=str, default='uint16', choices=['float32', 'uint16'],
                        help='TIF output dtype: float32 (lossless) / uint16 (matches original)')
    # ---- CFG 参数 ----
    parser.add_argument('--use_cfg', action='store_true', help='Enable CFG latent-space inference')
    parser.add_argument('--prompt_neg', type=str, default='cloudy, blurry, low quality', help='Negative prompt for CFG')
    parser.add_argument('--cfg_scales', nargs='+', type=float, default=[0.8, 0.9, 1.0, 1.1, 1.2],
                        help='CFG scales, e.g. --cfg_scales 0.8 1.0 1.2')
    parser.add_argument('--cfg_sequential', action='store_true',
                        help='CFG sequential mode: finish all images for one cfg before next (slower, allows early inspection)')
    args = parser.parse_args()

    if args.use_tif and not HAS_RASTERIO:
        raise RuntimeError("--use_tif requires rasterio, but it is not installed. Run: pip install rasterio")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Difix(
        pretrained_name=args.model_name,
        pretrained_path=None,
        timestep=args.timestep,
        mv_unet=args.mv_unet,
    )
    model = load_ckpt_from_state_dict_noopt(model, args.model_path)
    model.set_eval()
    print("Pretrained path:", args.model_path)

    if os.path.isdir(args.input_image):
        input_images = load_images(args.input_image)
    else:
        input_images = [args.input_image]

    if args.ref_image is not None:
        if os.path.isdir(args.ref_image):
            ref_images = load_images(args.ref_image)
        else:
            ref_images = [args.ref_image]
        assert len(input_images) == len(ref_images), "Input and reference counts must match."
    else:
        ref_images = [None] * len(input_images)

    prompt_dict = {}
    if os.path.exists("./prompt.txt"):
        with open("prompt.txt", "r") as f:
            for line in f:
                if ":" in line:
                    fname, extra = line.strip().split(":", 1)
                    prompt_dict[fname.strip()] = extra.strip()

    # 视频模式：提前打开 writer
    video_writer = None
    if args.video:
        video_path = os.path.join(args.output_dir, "output.mp4")
        video_writer = imageio.get_writer(video_path, fps=30)

    if args.use_cfg:
        if args.cfg_sequential:
            # ══════ CFG 串行：逐 cfg → 逐图（慢，可中间检查）══════
            for cfg in args.cfg_scales:
                cfg_str = f"cfg{str(cfg).replace('.', '_')}"
                cfg_subdir = os.path.join(args.output_dir, cfg_str)
                os.makedirs(cfg_subdir, exist_ok=True)
                print(f"\n{'='*50}\n  CFG = {cfg}  ({cfg_str})\n{'='*50}")

                for in_path, rf_path in tqdm(list(zip(input_images, ref_images)),
                                             desc=f"CFG={cfg}", total=len(input_images)):
                    fname = os.path.basename(in_path)
                    extra_prompt = prompt_dict.get(fname, "")
                    pos_prompt = f"{args.prompt}, {extra_prompt}" if extra_prompt else args.prompt
                    neg_prompt = f"{args.prompt_neg}, {extra_prompt}" if extra_prompt else args.prompt_neg

                    x = preprocess_like_training(in_path, rf_path, args.height, args.width, device,
                                                 use_tif=args.use_tif, sar_type=args.sar_type)
                    with torch.no_grad():
                        cfg_outputs = run_latent_cfg(model, x, pos_prompt, neg_prompt, [cfg])
                    y = cfg_outputs[cfg]
                    y_img = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                    y_scaled = y_img * 10000.0
                    save_name = fname.replace('_cloudy', '_pred')
                    save_tif(os.path.join(cfg_subdir, save_name), y_scaled, dtype=args.save_dtype)
                print(f"  CFG={cfg} done → {cfg_subdir}/")
        else:
            # ══════ CFG 并行：逐图 → 全 cfg 一次出（快，encode+UNet 复用）══════
            # 预建子目录
            cfg_dirs = {}
            for cfg in args.cfg_scales:
                cfg_str = f"cfg{str(cfg).replace('.', '_')}"
                cfg_subdir = os.path.join(args.output_dir, cfg_str)
                os.makedirs(cfg_subdir, exist_ok=True)
                cfg_dirs[cfg] = cfg_subdir

            for in_path, rf_path in tqdm(list(zip(input_images, ref_images)),
                                         desc="CFG inference", total=len(input_images)):
                fname = os.path.basename(in_path)
                extra_prompt = prompt_dict.get(fname, "")
                pos_prompt = f"{args.prompt}, {extra_prompt}" if extra_prompt else args.prompt
                neg_prompt = f"{args.prompt_neg}, {extra_prompt}" if extra_prompt else args.prompt_neg

                x = preprocess_like_training(in_path, rf_path, args.height, args.width, device,
                                             use_tif=args.use_tif, sar_type=args.sar_type)
                with torch.no_grad():
                    cfg_outputs = run_latent_cfg(model, x, pos_prompt, neg_prompt, args.cfg_scales)

                save_name = fname.replace('_cloudy', '_pred')
                for cfg, y in cfg_outputs.items():
                    y_img = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                    y_scaled = y_img * 10000.0
                    save_tif(os.path.join(cfg_dirs[cfg], save_name), y_scaled, dtype=args.save_dtype)
    else:
        # ══════ 标准模式 ══════
        for in_path, rf_path in tqdm(list(zip(input_images, ref_images)),
                                     desc="Processing images", total=len(input_images)):
            fname = os.path.basename(in_path)
            extra_prompt = prompt_dict.get(fname, "")
            final_prompt = args.prompt if extra_prompt == "" else f"{args.prompt}, {extra_prompt}"

            x = preprocess_like_training(in_path, rf_path, args.height, args.width, device,
                                         use_tif=args.use_tif, sar_type=args.sar_type)

            with torch.no_grad():
                y = model(x, prompt=final_prompt)

            y_img = (y[0, 0] * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
            y_scaled = y_img * 10000.0

            if args.video:
                frame = (y_scaled / 10000.0 * 255).clip(0, 255).astype(np.uint8)
                video_writer.append_data(frame)
            else:
                save_name = fname.replace('_cloudy', '_pred')
                out_path = os.path.join(args.output_dir, save_name)
                save_tif(out_path, y_scaled, dtype=args.save_dtype)

    if video_writer is not None:
        video_writer.close()

