import os
import time
import torch
import argparse
from tqdm import tqdm
from torch.hub import download_url_to_file

from utils.load_utils import init_create_model
from pipelines.inference import RelightVidInference
from utils.video_utils import *


def main(args):
    # Load model config and model
    relvid_model = init_create_model(args.config_path)
    relvid_model = relvid_model.to('cuda').half()

    # Setup inference pipeline
    inf_pipe = RelightVidInference(
        relvid_model.unet,
        scheduler='ddpm',
        num_ddim_steps=20
    )

    # Prepare input tensors
    fg_tensor = apply_mask_to_video(args.input, args.mask).cuda().unsqueeze(0).to(dtype=torch.float16)
    bg_tensor = load_and_process_video(args.bg_cond).cuda().unsqueeze(0).to(dtype=torch.float16)

    # bg_tensor = bg_tensor.clamp(0, 1)
    fg_tensor = fg_tensor[:, :, :, 280:(1280 - 280), :] # clip to 1280 x 720
    bg_tensor = bg_tensor[:, :, :, 280:(1280 - 280), :]
    print(fg_tensor.size(), bg_tensor.size())

    threshold = -0.9 
    # 如果 R, G, B 三个通道都小于阈值，则将该像素点替换为 0 (灰色)
    # 在 Color 维度（dim=2）上进行操作
    mask = (bg_tensor < threshold).all(dim=2, keepdim=True)
    bg_tensor.masked_fill_(mask, 0)
    

    cond_fg_tensor = relvid_model.encode_image_to_latent(fg_tensor)
    cond_bg_tensor = relvid_model.encode_image_to_latent(bg_tensor)
    cond_tensor = torch.cat((cond_fg_tensor, cond_bg_tensor), dim=2)

    # Initialize latent tensor
    init_latent = torch.randn_like(cond_fg_tensor)

    # Text conditioning
    text_cond = relvid_model.encode_text([args.prompt])
    text_uncond = relvid_model.encode_text([''])

    # Convert to float16
    init_latent, text_cond, text_uncond, cond_tensor = (
        init_latent.to(dtype=torch.float16),
        text_cond.to(dtype=torch.float16),
        text_uncond.to(dtype=torch.float16),
        cond_tensor.to(dtype=torch.float16)
    )
    inf_pipe.unet.to(torch.float16)

    # Inference
    latent_pred = inf_pipe(
        latent=init_latent,
        text_cond=text_cond,
        text_uncond=text_uncond,
        img_cond=cond_tensor,
        text_cfg=7.5,
        img_cfg=1.2,
    )['latent']

    # Decode and save video
    image_pred = relvid_model.decode_latent_to_image(latent_pred)
    save_video_from_frames(image_pred, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./assets/input/woman.mp4")
    parser.add_argument("--mask", type=str, default="./assets/mask/woman")
    parser.add_argument("--bg_cond", type=str, default="./assets/video_bg/universe1.mp4")
    
    parser.add_argument("--config_path", type=str, default="configs/inference_fbc.yaml")
    parser.add_argument("--output_path", type=str, default="output/woman_universe1.mp4")
    parser.add_argument("--prompt", type=str, default="change the background")

    args = parser.parse_args()
    main(args)
