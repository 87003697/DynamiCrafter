import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# Import the new DynamiCrafter guidance pipeline
from guidance_pipeline import DynamiCrafterGuidancePipeline

# Enhanced device selection - support specific CUDA devices
def get_device():
    """Get the appropriate CUDA device."""
    # Check if CUDA_VISIBLE_DEVICES is set
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None:
        # Use first visible device
        device_num = cuda_visible.split(',')[0]
        return torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    else:
        # Default to cuda:3 if available, otherwise first available CUDA device
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 3:
                return torch.device("cuda:3")
            else:
                return torch.device("cuda:0")
        else:
            return torch.device("cpu")

device = get_device()
print(f"[DEBUG] Using device: {device}")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_parser():
    """Create argument parser for DynamiCrafter guidance generation."""
    parser = argparse.ArgumentParser(description="DynamiCrafter Guidance Pipeline Generation")
    
    # Basic parameters
    parser.add_argument("--prompt", type=str, 
                       default="A person walking in a beautiful garden with flowers blooming, cinematic lighting",
                       help="Text prompt for video generation")
    parser.add_argument("--condition_image", type=str, 
                       default="prompts/1024/pour_bear.png",
                       help="Path to input condition image")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    # DynamiCrafter specific parameters
    parser.add_argument("--resolution", type=str, default="256_256", choices=["256_256", "512_512", "1024_1024"],
                       help="Model resolution")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to generate")
    parser.add_argument("--frame_stride", type=int, default=3,
                       help="Frame stride parameter")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="DynamiCrafter guidance scale")
    parser.add_argument("--eta", type=float, default=0.0,
                       help="DDIM eta parameter")

    # Guidance optimization parameters
    parser.add_argument("--num_optimization_steps", type=int, default=1000,
                       help="Number of optimization steps")
    parser.add_argument("--learning_rate", type=float, default=0.05,
                       help="Learning rate for optimization")
    parser.add_argument("--loss_type", type=str, default="sds", choices=["sds", "csd", "rfds"], 
                       help="Loss function type: sds (traditional), csd (classifier-free), or rfds (rectified flow distillation)")
    parser.add_argument("--weight_type", type=str, default="auto", choices=["auto", "t", "ada", "uniform"], 
                       help="Weighting strategy: auto (default for each loss), t (time-dependent), ada (adaptive), uniform (no weighting)")
    parser.add_argument("--cfg_scale", type=float, default=7.5,
                       help="CFG scale for guidance loss")
    parser.add_argument("--optimizer_type", type=str, default="AdamW", choices=["AdamW", "Adam"],
                       help="Optimizer type")

    # Dynamic step ratio parameters
    parser.add_argument("--min_step_ratio_start", type=float, default=0.02,
                       help="Starting minimum step ratio")
    parser.add_argument("--min_step_ratio_end", type=float, default=0.02,
                       help="Ending minimum step ratio")
    parser.add_argument("--max_step_ratio_start", type=float, default=0.98,
                       help="Starting maximum step ratio")
    parser.add_argument("--max_step_ratio_end", type=float, default=0.98,
                       help="Ending maximum step ratio")

    # Enhanced saving parameters
    parser.add_argument("--save_results", action="store_true", 
                       help="Enable enhanced result saving with organized directory structure")
    parser.add_argument("--results_dir", type=str, default="results_dynamicrafter_guidance",
                       help="Base directory for saving results")
    parser.add_argument("--save_debug_images", action="store_true", 
                       help="Save debug frame images during optimization")
    parser.add_argument("--save_debug_videos", action="store_true", 
                       help="Save debug videos during optimization")
    parser.add_argument("--save_process_video", action="store_true", 
                       help="Create optimization process video")
    parser.add_argument("--debug_save_interval", type=int, default=100, 
                       help="Save debug results every N steps")

    # Misc parameters
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt for guidance")
    parser.add_argument("--output_type", type=str, default="tensor", choices=["tensor", "numpy"],
                       help="Output format")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., 'cuda:3', 'cuda:0', 'cpu'). If not specified, defaults to cuda:3")
    
    return parser


def main():
    """Main function for DynamiCrafter guidance generation."""
    parser = create_parser()
    args = parser.parse_args()

    # Enhanced device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"[INFO] Using device: {device}")

    # Set seed
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"[INFO] Starting DynamiCrafter guidance generation...")
    print(f"[INFO] Resolution: {args.resolution}")
    print(f"[INFO] Loss type: {args.loss_type}")
    print(f"[INFO] Weight type: {args.weight_type}")
    print(f"[INFO] Optimization steps: {args.num_optimization_steps}")
    print(f"[INFO] Learning rate: {args.learning_rate}")
    print(f"[INFO] CFG scale: {args.cfg_scale}")

    # --- 1. Setup Pipeline ---
    print("[INFO] Loading DynamiCrafter Guidance Pipeline...")
    pipeline = DynamiCrafterGuidancePipeline(
        resolution=args.resolution,
        device=str(device)  # Pass device as string
    )
    print("[INFO] Pipeline loaded successfully")

    # --- 2. Load and process input image ---
    print(f"[INFO] Loading condition image from: {args.condition_image}")
    if not os.path.exists(args.condition_image):
        print(f"[ERROR] Condition image not found: {args.condition_image}")
        return
    
    input_image = Image.open(args.condition_image).convert("RGB")
    print(f"[DEBUG] Loaded input image: {input_image.size}")

    # --- 3. Setup output directory ---
    # Truncate prompt for directory name to avoid OSError: File name too long
    prompt_str_for_path = "".join(filter(str.isalnum, args.prompt))[:50]

    # Determine effective weight type
    if args.weight_type == "auto":
        if args.loss_type == "sds":
            effective_weight_type = "t"
        elif args.loss_type == "csd":
            effective_weight_type = "ada"
        else:  # rfds
            effective_weight_type = "uniform"
    else:
        effective_weight_type = args.weight_type

    # Create descriptive save directory name
    loss_desc = f"{args.loss_type}_{effective_weight_type}"

    if args.output_dir is None:
        save_dir = "outputs/dynamicrafter_guidance_pipeline/%s_%s_lr%.3f_seed%d_cfg%.1f_steps%d" % (
            prompt_str_for_path,
            loss_desc,
            args.learning_rate,
            args.seed,
            args.cfg_scale,
            args.num_optimization_steps
        )
    else:
        save_dir = args.output_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Save directory: {save_dir}")

    # Save condition image for reference
    input_image.save(os.path.join(save_dir, "condition_image.png"))

    # --- 4. Run Generation ---
    print(f"[INFO] Starting generation with {args.loss_type} loss...")
    print(f"[INFO] Using {args.loss_type.upper()}-{effective_weight_type.upper()} loss")

    start_time = time.time()
    
    result = pipeline(
        # Basic parameters
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        generator=generator,
        
        # Guidance parameters
        num_optimization_steps=args.num_optimization_steps,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        weight_type=args.weight_type,
        cfg_scale=args.cfg_scale,
        optimizer_type=args.optimizer_type,
        
        # Dynamic step ratio parameters
        min_step_ratio_start=args.min_step_ratio_start,
        min_step_ratio_end=args.min_step_ratio_end,
        max_step_ratio_start=args.max_step_ratio_start,
        max_step_ratio_end=args.max_step_ratio_end,
        
        # Enhanced saving parameters
        save_results=args.save_results,
        results_dir=args.results_dir,
        save_debug_images=args.save_debug_images,
        save_debug_videos=args.save_debug_videos,
        save_process_video=args.save_process_video,
        debug_save_interval=args.debug_save_interval,
        
        # Output parameters
        output_type=args.output_type,
        return_dict=True,
    )

    elapsed_time = time.time() - start_time
    print(f"[INFO] Generation completed successfully in {elapsed_time:.2f} seconds!")

    # --- 5. Save Results ---
    print("[INFO] Saving final results...")
    
    # Save final video using pipeline's save function
    final_videos = result["videos"]
    final_video_path = os.path.join(save_dir, "final_video.mp4")
    pipeline.save_video(final_videos, final_video_path, fps=8)
    print(f"[INFO] Final video saved to: {final_video_path}")

    # Check for NaN values
    if torch.isnan(final_videos).any():
        print("[WARNING] NaN values detected in generated video!")
    else:
        print("[INFO] Video generation successful - no NaN values detected")

    # Save parameters for reference
    params_file = os.path.join(save_dir, "generation_parameters.txt")
    with open(params_file, "w") as f:
        f.write("DynamiCrafter Guidance Generation Parameters\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generation time: {elapsed_time:.2f} seconds\n\n")
        
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nEffective weight type: {effective_weight_type}\n")
        f.write(f"Video shape: {final_videos.shape}\n")

    print(f"[INFO] Parameters saved to: {params_file}")

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"✅ Model: DynamiCrafter {args.resolution}")
    print(f"✅ Loss type: {args.loss_type.upper()}-{effective_weight_type.upper()}")
    print(f"✅ Optimization steps: {args.num_optimization_steps}")
    print(f"✅ Learning rate: {args.learning_rate}")
    print(f"✅ Video shape: {final_videos.shape}")
    print(f"✅ Generation time: {elapsed_time:.2f} seconds")
    print(f"✅ Results saved to: {save_dir}")
    
    if args.save_results:
        print(f"✅ Enhanced saving enabled")
        if args.save_debug_images:
            print("  • Debug images saved")
        if args.save_debug_videos:
            print("  • Debug videos saved")
        if args.save_process_video:
            print("  • Process video generated")

    print("✅ Script finished successfully!")


if __name__ == "__main__":
    main() 