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
    # Add the VAR directory to path first
    import sys
    import os
    var_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VAR')
    if var_path not in sys.path:
        sys.path.insert(0, var_path)
    
    from models import VQVAE, build_vae_var  # From VAR repository
    print("Successfully imported VAR models")
except ImportError as e:
    print(f"Warning: Could not import VAR models: {e}")
    build_vae_var = None


def load_var_models(var_checkpoint_dir, model_depth=24, device_for_loading='cuda', pipeline_target_device='cuda'):
    """
    Load VAR model and VAE
    
    Args:
        var_checkpoint_dir: Directory containing VAR checkpoints
        model_depth: Model depth (16, 20, 24, 30, or 36)
        device_for_loading: Device to load models on
        pipeline_target_device: Target device for the pipeline (for flash attention decision)
    
    Returns:
        vae, var: Loaded VAE and VAR models
    """
    if build_vae_var is None:
        print("Warning: VAR models not available. Skipping VAR model loading.")
        return None, None
    
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
        print(f"Warning: VAE checkpoint not found at {local_vae_ckpt_path}")
        return None, None
    if not osp.exists(local_var_ckpt_path):
        print(f"Warning: VAR checkpoint not found at {local_var_ckpt_path}")
        return None, None
    
    # Use 256px patch configuration for all models (this is what works)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    # Determine if flash attention should be used for the VAR model
    flash_attention_for_var = True
    if pipeline_target_device == 'mps':
        flash_attention_for_var = False
        print("Pipeline target device is MPS, explicitly disabling flash attention for VAR model.")
    elif device_for_loading == 'cpu' and pipeline_target_device == 'cpu': # If VAR is on CPU and pipeline is on CPU
        flash_attention_for_var = False
        print("VAR model and pipeline on CPU, explicitly disabling flash attention for VAR model.")
    else:
        # Default case, usually for CUDA where flash attention is beneficial
        print(f"VAR model flash attention decision: {flash_attention_for_var} (pipeline: {pipeline_target_device}, loading: {device_for_loading})")


    try:
        # Build VAE and VAR models with working configuration
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=device_for_loading, patch_nums=patch_nums,
            num_classes=1000, depth=model_depth, shared_aln=False,
            flash_if_available=flash_attention_for_var  # Pass the flag here
        )
        
        # Load checkpoints
        map_location = 'cpu' if device_for_loading == 'cpu' else device_for_loading
        vae.load_state_dict(torch.load(local_vae_ckpt_path, map_location=map_location), strict=True)
        var_state_dict = torch.load(local_var_ckpt_path, map_location=map_location)
        var.load_state_dict(var_state_dict, strict=False)  # Use strict=False for potential mismatches
        
        print(f"Successfully loaded VAR model with 256px patch configuration. Flash attention: {flash_attention_for_var}")
        
        # Move models to the correct device after loading
        vae = vae.to(device_for_loading)
        var = var.to(device_for_loading)
        
        # For CPU execution, ensure compatibility (this might be redundant if flash_if_available=False was set)
        if device_for_loading == 'cpu' and not flash_attention_for_var:
            print("VAR model configured for CPU execution (flash attention disabled).")
        elif device_for_loading == 'cpu': # Original block, may need review
            try:
                # Attempt to disable xformers if still necessary, though flash_if_available should handle it
                # from xformers import ops
                # var.set_use_memory_efficient_attention_xformers(False, attention_op=None)
                print("Configuring VAR model for CPU execution (post-load check)...")
            except Exception as e:
                print(f"Could not disable xformers for VAR on CPU post-load: {e}")

        # Set to eval mode
        vae.eval()
        var.eval()
        
        # Disable gradients
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in var.parameters():
            p.requires_grad_(False)
        
        print(f'VAR model preparation finished. Model depth: {model_depth}, Device: {device_for_loading}')
        return vae, var
        
    except Exception as e:
        print(f"Error loading VAR models: {e}")
        print("Continuing without VAR models - will use diffusion for candidate image generation")
        return None, None


def main(args):
    torch.set_grad_enabled(False)
    # On macOS, CUDA is not available, so use CPU and MPS if available
    if torch.backends.mps.is_available():
        pipeline_device = "mps"
        dtype = torch.float32  # MPS works better with float32
        print("Using MPS (Metal Performance Shaders) on macOS")
    else:
        pipeline_device = "cpu"
        dtype = torch.float32  # CPU uses float32
        print("Using CPU on macOS")
    
    # Load VAR models if checkpoint directory is provided
    var_vae, var_model = None, None
    if hasattr(args, 'var_checkpoint_dir') and args.var_checkpoint_dir:
        print("Loading VAR models...")
        var_model_depth = getattr(args, 'var_model_depth', 24)
        
        # For macOS, always use CPU for VAR models initial loading
        var_device_for_loading = "cpu"
        print(f"Loading VAR models on device: {var_device_for_loading} (optimized for macOS)")
        
        var_vae, var_model = load_var_models(
            args.var_checkpoint_dir, 
            model_depth=var_model_depth, 
            device_for_loading=var_device_for_loading,
            pipeline_target_device=pipeline_device # Pass the determined pipeline device
        )
    
    sd_path = '/Users/gilliam/Desktop/493G1/VideoVAR/stable-diffusion-v1-4'
    unet = get_models(args, sd_path).to(pipeline_device, dtype=dtype)  # Use pipeline_device and dtype
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=dtype).to(pipeline_device)  # Use dtype and pipeline_device
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=dtype).to(pipeline_device)  # Use dtype and pipeline_device
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
    ).to(pipeline_device)
    
    videogen_pipeline.image_reward_model = image_reward_model
    # Don't enable xformers on macOS as it's not supported
    if torch.cuda.is_available():
        videogen_pipeline.enable_xformers_memory_efficient_attention()
        print("Enabled xformers memory efficient attention")
    else:
        print("Xformers not enabled (running on macOS/CPU)")

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