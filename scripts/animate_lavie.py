import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.split(sys.path[0])[0])

import torch
import argparse
import torchvision
import random
import numpy as np

from i4vgen.lavie.pipelines.pipeline_videogen import VideoGenPipeline

from i4vgen.lavie.utils.download import find_model
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from omegaconf import OmegaConf

from i4vgen.lavie.models import get_models
import imageio

from einops import rearrange
from torchvision.utils import save_image
import ImageReward as RM

# VAR imports
import os.path as osp
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

# Add VAR model imports (you may need to adjust the import path based on your setup)
try:
    from VAR.models import VQVAE, build_vae_var  # From VAR repository
except ImportError:
    print("Warning: Could not import VAR models. Make sure VAR repository is in your Python path.")
    build_vae_var = None


def load_var_models(var_checkpoint_dir, model_depth=24, device='cuda'):
    """
    Load VAR model and VAE
    
    Args:
        var_checkpoint_dir: Directory containing VAR checkpoints
        model_depth: Model depth (16, 20, 24, 30, or 36)
        device: Device to load models on
    
    Returns:
        vae, var: Loaded VAE and VAR models
    """
    if build_vae_var is None:
        raise ImportError("VAR models not available. Please ensure VAR repository is properly installed.")
    
    # Disable default parameter init for faster speed (from VAR example)
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    
    assert model_depth in {16, 20, 24, 30, 36}
    
    # Checkpoint paths
    vae_ckpt_name = 'vae_ch160v4096z32.pth'
    var_ckpt_name = f'var_d{model_depth}.pth'
    
    local_vae_ckpt_path = osp.join(var_checkpoint_dir, vae_ckpt_name)
    local_var_ckpt_path = osp.join(var_checkpoint_dir, var_ckpt_name)
    
    # Check if checkpoints exist
    if not osp.exists(local_vae_ckpt_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {local_vae_ckpt_path}")
    if not osp.exists(local_var_ckpt_path):
        raise FileNotFoundError(f"VAR checkpoint not found at {local_var_ckpt_path}")
    
    # Build VAE and VAR models
    FOR_512_px = model_depth == 30
    if FOR_512_px:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=FOR_512_px,
    )
    
    # Load checkpoints
    vae.load_state_dict(torch.load(local_vae_ckpt_path, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(local_var_ckpt_path, map_location='cpu'), strict=True)
    
    # Set to eval mode
    vae.eval()
    var.eval()
    
    # Disable gradients
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    
    print(f'VAR model preparation finished. Model depth: {model_depth}')
    
    return vae, var


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load VAR models if checkpoint directory is provided
    var_vae, var_model = None, None
    if hasattr(args, 'var_checkpoint_dir') and args.var_checkpoint_dir:
        print("Loading VAR models...")
        var_model_depth = getattr(args, 'var_model_depth', 24)
        var_vae, var_model = load_var_models(
            args.var_checkpoint_dir, 
            model_depth=var_model_depth, 
            device=device
        )
	
    sd_path = '/Users/gilliam/Desktop/493G1/VideoVAR/stable-diffusion-v1-4'
    unet = get_models(args, sd_path).to(device, dtype=torch.float16)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device) # huge
    image_reward_model = RM.load("ImageReward-v1.0")

    # set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()

    if args.sample_method == 'ddim':
        scheduler = DDIMScheduler.from_pretrained(sd_path, 
			subfolder="scheduler",
			beta_start=args.beta_start, 
			beta_end=args.beta_end, 
			beta_schedule=args.beta_schedule)
    elif args.sample_method == 'eulerdiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(sd_path,
			subfolder="scheduler",
			beta_start=args.beta_start,
			beta_end=args.beta_end,
			beta_schedule=args.beta_schedule)
    elif args.sample_method == 'ddpm':
        scheduler = DDPMScheduler.from_pretrained(sd_path,
			subfolder="scheduler",
			beta_start=args.beta_start,
			beta_end=args.beta_end,
			beta_schedule=args.beta_schedule)
    else:
        raise NotImplementedError
    
    # Create pipeline with VAR models
    videogen_pipeline = VideoGenPipeline(
        vae=vae, 
        text_encoder=text_encoder_one, 
        tokenizer=tokenizer_one, 
        scheduler=scheduler, 
        unet=unet,
        var_model=var_model,    # Add VAR model
        var_vae=var_vae         # Add VAR VAE
    ).to(device)
    
    videogen_pipeline.image_reward_model = image_reward_model
    videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Get ImageNet class for VAR generation
    imagenet_class = getattr(args, 'imagenet_class', 980)  # Default to class 980 (volcano)

    for i, cur_seed in enumerate(args.seed):
        # Set seeds for reproducibility
        torch.manual_seed(cur_seed)
        random.seed(cur_seed)
        np.random.seed(cur_seed)
        
        for prompt in args.text_prompt:
            print('Processing the ({}) prompt with seed {}'.format(prompt, cur_seed))
            print(f'Using ImageNet class {imagenet_class} for VAR image generation')
            
            sample = videogen_pipeline(
                prompt=prompt, 
                video_length=args.video_length, 
                height=args.image_size[0], 
                width=args.image_size[1], 
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                imagenet_class=imagenet_class  # Add ImageNet class parameter
            )
            
            videos = sample.video
            video_mp4_name = args.output_folder + prompt.replace(' ', '_') + '-{}.mp4'.format(cur_seed)
            video_mp4 = videos[0]
            torchvision.io.write_video(video_mp4_name, video_mp4, fps=8)
            print(f'Saved video: {video_mp4_name}')

            # Save candidate images if requested
            if getattr(args, 'save_candidate_images', False):
                candidate_images = sample.candidate_images
                video_png = rearrange(candidate_images, "b f h w c -> (b f) c h w").contiguous()
                video_png = video_png.float()
                video_png = video_png / 255.
                png_name = args.output_folder + prompt.replace(' ', '_') + '-{}-candidate-images.jpg'.format(cur_seed)
                save_image(video_png, png_name, nrow=4)
                print(f'Saved candidate images: {png_name}')

            # Save NI-VSDS video if requested
            if getattr(args, 'save_ni_vsds_video', False):
                ni_vsds_video = sample.ni_vsds_video
                video_mp4_name = args.output_folder + prompt.replace(' ', '_') + '-{}-ni-vsds-video.mp4'.format(cur_seed)
                video_mp4 = ni_vsds_video[0]
                torchvision.io.write_video(video_mp4_name, video_mp4, fps=8)
                print(f'Saved NI-VSDS video: {video_mp4_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument("--var_checkpoint_dir", type=str, default="", help="Directory containing VAR checkpoints")
    parser.add_argument("--var_model_depth", type=int, default=24, choices=[16, 20, 24, 30, 36], help="VAR model depth")
    parser.add_argument("--imagenet_class", type=int, default=980, help="ImageNet class for VAR generation")
    parser.add_argument("--model", type=str, default="UNet", help="Model type to use")  # Add this line
    args = parser.parse_args()

    if args.config:
        config_args = OmegaConf.load(args.config)
        # Override with command line arguments if provided
        if args.var_checkpoint_dir:
            config_args.var_checkpoint_dir = args.var_checkpoint_dir
        if args.var_model_depth != 24:  # Only override if different from default
            config_args.var_model_depth = args.var_model_depth
        if args.imagenet_class != 980:  # Only override if different from default
            config_args.imagenet_class = args.imagenet_class
        main(config_args)
    else:
        main(args)