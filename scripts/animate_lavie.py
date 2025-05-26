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
import numpy as np
from PIL import Image

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

# Add VAR model imports
try:
    # Add the VAR directory to path first
    import sys
    import os
    var_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VAR')
    if var_path not in sys.path:
        sys.path.insert(0, var_path)
    
    from models import VQVAE, build_vae_var
    print("Successfully imported VAR models")
except ImportError as e:
    print(f"Warning: Could not import VAR models: {e}")
    build_vae_var = None


def load_var_models(var_checkpoint_dir, model_depth=24, device='cpu', dtype=torch.float32):
    """
    Load VAR model and VAE for Mac compatibility
    Note: For MPS, we keep models on CPU to avoid dtype issues
    
    Args:
        var_checkpoint_dir: Directory containing VAR checkpoints
        model_depth: Model depth (16, 20, 24, 30, or 36)
        device: Device to load models on (for MPS, we'll use CPU)
        dtype: Data type for models
    
    Returns:
        vae, var: Loaded VAE and VAR models
    """
    if build_vae_var is None:
        print("Warning: VAR models not available. Skipping VAR model loading.")
        return None, None
    
    # For MPS, always use CPU for VAR to avoid dtype issues
    if device == 'mps':
        print("Note: Loading VAR models on CPU to avoid MPS dtype issues")
        var_device = 'cpu'
    else:
        var_device = device
    
    # Disable default parameter init for faster speed
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
    
    # Use 256px patch configuration
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    # Disable flash attention for Mac
    flash_attention = False
    print(f"Building VAR models with flash attention disabled for Mac compatibility")

    try:
        # Build VAE and VAR models on CPU first
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device='cpu', patch_nums=patch_nums,
            num_classes=1000, depth=model_depth, shared_aln=False,
            flash_if_available=False  # Explicitly disable flash attention for Mac
        )
        
        # Load checkpoints to CPU first
        vae.load_state_dict(torch.load(local_vae_ckpt_path, map_location='cpu'), strict=True)
        var_state_dict = torch.load(local_var_ckpt_path, map_location='cpu')
        var.load_state_dict(var_state_dict, strict=False)
        
        print(f"Successfully loaded VAR model checkpoints")
        
        # Convert to specified dtype
        vae = vae.to(dtype=dtype)
        var = var.to(dtype=dtype)
        
        # Move to target device (CPU for MPS to avoid issues)
        if var_device != 'cpu':
            vae = vae.to(var_device)
            var = var.to(var_device)
            print(f"Moved VAR models to {var_device}")
        else:
            print("Keeping VAR models on CPU")
        
        # Set to eval mode
        vae.eval()
        var.eval()
        
        # Disable gradients
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in var.parameters():
            p.requires_grad_(False)
        
        print(f'VAR model preparation finished. Model depth: {model_depth}, Device: {var_device}')
        return vae, var
        
    except Exception as e:
        print(f"Error loading VAR models: {e}")
        print("Continuing without VAR models - will use diffusion for candidate image generation")
        return None, None


def main(args):
    torch.set_grad_enabled(False)
    
    # Device and dtype selection for Mac
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS works better with float32
        print("Using MPS (Metal Performance Shaders) on macOS")
        # Set default dtype to float32 for MPS compatibility
        torch.set_default_dtype(torch.float32)
    else:
        device = "cpu"
        dtype = torch.float32
        print("Using CPU on macOS")
    
    # Load VAR models if checkpoint directory is provided
    var_vae, var_model = None, None
    if hasattr(args, 'var_checkpoint_dir') and args.var_checkpoint_dir:
        print("Loading VAR models...")
        var_model_depth = getattr(args, 'var_model_depth', 24)
        
        # Load VAR models with Mac compatibility
        var_vae, var_model = load_var_models(
            args.var_checkpoint_dir, 
            model_depth=var_model_depth,
            device=device,
            dtype=dtype
        )
    
    # Paths - update this to your actual path
    sd_path = args.sd_path if hasattr(args, 'sd_path') else '/Users/gilliam/Desktop/493G1/VideoVAR/stable-diffusion-v1-4'
    
    # Load models with consistent dtype
    unet = get_models(args, sd_path).to(device, dtype=dtype)
    state_dict = find_model(args.ckpt_path)
    unet.load_state_dict(state_dict)

    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae", torch_dtype=dtype).to(device)
    tokenizer_one = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder_one = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder", torch_dtype=dtype).to(device)
    
    # Load ImageReward model
    try:
        image_reward_model = RM.load("ImageReward-v1.0")
    except Exception as e:
        print(f"Warning: Could not load ImageReward model: {e}")
        image_reward_model = None

    # Set eval mode
    unet.eval()
    vae.eval()
    text_encoder_one.eval()

    # Configure scheduler
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
        raise NotImplementedError(f"Sample method {args.sample_method} not implemented")
    
    # Create pipeline with VAR models
    videogen_pipeline = VideoGenPipeline(
        vae=vae, 
        text_encoder=text_encoder_one, 
        tokenizer=tokenizer_one, 
        scheduler=scheduler, 
        unet=unet,
        var_model=var_model,
        var_vae=var_vae
    ).to(device)
    
    # Set image reward model if available
    if image_reward_model is not None:
        videogen_pipeline.image_reward_model = image_reward_model
    
    # Don't enable xformers on macOS
    print("Xformers not enabled (running on macOS)")

    # Create output directory
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Get ImageNet class for VAR generation
    imagenet_class = getattr(args, 'imagenet_class', 980)  # Default to class 980

    # Generate videos
    for i, cur_seed in enumerate(args.seed):
        # Set seeds for reproducibility
        torch.manual_seed(cur_seed)
        random.seed(cur_seed)
        np.random.seed(cur_seed)
        
        for prompt in args.text_prompt:
            print(f'Processing prompt: "{prompt}" with seed {cur_seed}')
            print(f'Using ImageNet class {imagenet_class} for VAR image generation')
            
            try:
                # Generate video
                with torch.no_grad():
                    # Don't use autocast for MPS to avoid dtype mixing issues
                    sample = videogen_pipeline(
                        prompt=prompt, 
                        video_length=args.video_length, 
                        height=args.image_size[0], 
                        width=args.image_size[1], 
                        num_inference_steps=args.num_sampling_steps,
                        guidance_scale=args.guidance_scale,
                        use_fp16=False,  # Disable fp16 for Mac
                        imagenet_class=imagenet_class
                    )
                
                # Save video with Mac-compatible codec
                videos = sample.video
                video_mp4_name = os.path.join(args.output_folder, f"{prompt.replace(' ', '_')}-{cur_seed}.mp4")
                video_mp4 = videos[0]
                
                # Save video with Mac-compatible methods
                video_saved = False
                
                # First, ensure video tensor is in correct format (T, H, W, C) with uint8
                if video_mp4.dtype != torch.uint8:
                    video_mp4 = video_mp4.clamp(0, 255).to(torch.uint8)
                
                # Try torchvision with available codecs
                codecs_to_try = ['h264_videotoolbox', 'mpeg4', 'libopenh264']
                for codec in codecs_to_try:
                    try:
                        torchvision.io.write_video(video_mp4_name, video_mp4, fps=8, video_codec=codec)
                        print(f'Saved video ({codec}): {video_mp4_name}')
                        video_saved = True
                        break
                    except Exception as e:
                        print(f'Failed to save with {codec}: {str(e)[:50]}...')
                        continue
                
                # If torchvision fails, try imageio
                if not video_saved:
                    try:
                        # Convert to numpy for imageio
                        video_np = video_mp4.numpy()
                        imageio.mimsave(video_mp4_name, video_np, fps=8, codec='libx264')
                        print(f'Saved video (imageio): {video_mp4_name}')
                        video_saved = True
                    except Exception as e:
                        print(f'Failed to save with imageio: {str(e)[:50]}...')
                
                # Final fallback: save as individual frames
                if not video_saved:
                    print(f"All video codecs failed, saving individual frames...")
                    frames_dir = os.path.join(args.output_folder, f"{prompt.replace(' ', '_')}-{cur_seed}_frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    for frame_idx, frame in enumerate(video_mp4):
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                        # Convert frame to PIL Image and save
                        frame_np = frame.numpy().astype(np.uint8)
                        frame_pil = Image.fromarray(frame_np)
                        frame_pil.save(frame_path)
                    print(f'Saved video frames to: {frames_dir}')

                # Save candidate images if requested
                if getattr(args, 'save_candidate_images', False):
                    candidate_images = sample.candidate_images
                    video_png = rearrange(candidate_images, "b f h w c -> (b f) c h w").contiguous()
                    video_png = video_png.float() / 255.
                    png_name = os.path.join(args.output_folder, f"{prompt.replace(' ', '_')}-{cur_seed}-candidate-images.jpg")
                    save_image(video_png, png_name, nrow=4)
                    print(f'Saved candidate images: {png_name}')

                # Save NI-VSDS video if requested
                if getattr(args, 'save_ni_vsds_video', False):
                    ni_vsds_video = sample.ni_vsds_video
                    video_mp4_name = os.path.join(args.output_folder, f"{prompt.replace(' ', '_')}-{cur_seed}-ni-vsds-video.mp4")
                    video_mp4 = ni_vsds_video[0]
                    
                    # Save NI-VSDS video with Mac-compatible methods
                    ni_vsds_video_saved = False
                    
                    # First, ensure video tensor is in correct format (T, H, W, C) with uint8
                    if video_mp4.dtype != torch.uint8:
                        video_mp4 = video_mp4.clamp(0, 255).to(torch.uint8)
                    
                    # Try torchvision with available codecs
                    codecs_to_try = ['h264_videotoolbox', 'mpeg4', 'libopenh264']
                    for codec in codecs_to_try:
                        try:
                            torchvision.io.write_video(video_mp4_name, video_mp4, fps=8, video_codec=codec)
                            print(f'Saved NI-VSDS video ({codec}): {video_mp4_name}')
                            ni_vsds_video_saved = True
                            break
                        except Exception as e:
                            print(f'Failed to save NI-VSDS video with {codec}: {str(e)[:50]}...')
                            continue
                    
                    # If torchvision fails, try imageio
                    if not ni_vsds_video_saved:
                        try:
                            # Convert to numpy for imageio
                            video_np = video_mp4.numpy()
                            imageio.mimsave(video_mp4_name, video_np, fps=8, codec='libx264')
                            print(f'Saved NI-VSDS video (imageio): {video_mp4_name}')
                            ni_vsds_video_saved = True
                        except Exception as e:
                            print(f'Failed to save NI-VSDS video with imageio: {str(e)[:50]}...')
                    
                    # Final fallback: save as individual frames
                    if not ni_vsds_video_saved:
                        print(f"All video codecs failed for NI-VSDS video, saving individual frames...")
                        frames_dir = os.path.join(args.output_folder, f"{prompt.replace(' ', '_')}-{cur_seed}-ni-vsds_frames")
                        os.makedirs(frames_dir, exist_ok=True)
                        for frame_idx, frame in enumerate(video_mp4):
                            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
                            # Convert frame to PIL Image and save
                            frame_np = frame.numpy().astype(np.uint8)
                            frame_pil = Image.fromarray(frame_np)
                            frame_pil.save(frame_path)
                        print(f'Saved NI-VSDS video frames to: {frames_dir}')
                    
            except Exception as e:
                print(f"Error generating video for prompt '{prompt}': {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--sd_path", type=str, default="/Users/gilliam/Desktop/493G1/VideoVAR/stable-diffusion-v1-4", help="Path to Stable Diffusion model")
    parser.add_argument("--var_checkpoint_dir", type=str, default="", help="Directory containing VAR checkpoints")
    parser.add_argument("--var_model_depth", type=int, default=24, choices=[16, 20, 24, 30, 36], help="VAR model depth")
    parser.add_argument("--imagenet_class", type=int, default=980, help="ImageNet class for VAR generation")
    parser.add_argument("--model", type=str, default="UNet", help="Model type to use")
    args = parser.parse_args()

    # Load config
    config_args = OmegaConf.load(args.config)
    
    # Override with command line arguments
    if args.var_checkpoint_dir:
        config_args.var_checkpoint_dir = args.var_checkpoint_dir
    if args.sd_path:
        config_args.sd_path = args.sd_path
    config_args.var_model_depth = args.var_model_depth
    config_args.imagenet_class = args.imagenet_class
    config_args.model = args.model
    
    main(config_args)