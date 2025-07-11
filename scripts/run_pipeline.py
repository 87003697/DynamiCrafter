#!/usr/bin/env python3
"""
DynamiCrafter Pipeline Runner Script
Similar to run.sh but uses the DynamiCrafterImg2VideoPipeline class
"""

import os
import sys
import argparse
import glob
import time
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.gradio.dynamicrafter_pipeline import DynamiCrafterImg2VideoPipeline


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DynamiCrafter Pipeline Runner')
    
    # Main arguments
    parser.add_argument('resolution', type=str, choices=['256', '512', '1024'], 
                       help='Resolution version (256, 512, or 1024)')
    parser.add_argument('--seed', type=int, default=123, 
                       help='Random seed (default: 123)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    
    # Generation parameters
    parser.add_argument('--num_inference_steps', type=int, default=50,
                       help='Number of DDIM steps (default: 50)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Unconditional guidance scale (default: 7.5)')
    parser.add_argument('--eta', type=float, default=1.0,
                       help='DDIM eta parameter (default: 1.0)')
    parser.add_argument('--video_length', type=int, default=16,
                       help='Video length in frames (default: 16)')
    
    # Input/Output
    parser.add_argument('--prompt_dir', type=str, default=None,
                       help='Directory containing prompt images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--prompt_text', type=str, default="",
                       help='Text prompt for generation')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Single image path for generation')
    
    # Advanced options
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--fps', type=int, default=8,
                       help='Output video FPS (default: 8)')
    
    return parser.parse_args()


def get_resolution_config(resolution):
    """Get configuration for different resolutions"""
    if resolution == '256':
        return {
            'height': 256,
            'width': 256,
            'frame_stride': 3,
            'resolution_str': '256_256'
        }
    elif resolution == '512':
        return {
            'height': 320,
            'width': 512,
            'frame_stride': 24,
            'resolution_str': '512_512'
        }
    elif resolution == '1024':
        return {
            'height': 576,
            'width': 1024,
            'frame_stride': 10,
            'resolution_str': '1024_1024'
        }
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def find_image_files(prompt_dir):
    """Find all image files in the prompt directory"""
    if not os.path.exists(prompt_dir):
        print(f"❌ Prompt directory not found: {prompt_dir}")
        return []
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(prompt_dir, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(prompt_dir, f'*{ext.upper()}')))
    
    return sorted(image_files)


def load_text_prompt(image_path):
    """Load text prompt from corresponding .txt file"""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ""


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed
    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Get resolution configuration
    config = get_resolution_config(args.resolution)
    
    # Setup paths
    project_root = os.path.join(os.path.dirname(__file__), '..')
    
    # Default prompt directory
    if args.prompt_dir is None:
        args.prompt_dir = os.path.join(project_root, f'prompts/{args.resolution}/')
    
    # Default output directory
    if args.output_dir is None:
        name = f"dynamicrafter_pipeline_{args.resolution}_seed{args.seed}"
        args.output_dir = os.path.join(project_root, f'results/{name}/')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎬 DynamiCrafter Pipeline Runner")
    print("=" * 60)
    print(f"📐 Resolution: {args.resolution} ({config['height']}x{config['width']})")
    print(f"🎲 Seed: {args.seed}")
    print(f"💻 Device: {args.device}")
    print(f"📁 Prompt directory: {args.prompt_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🔧 DDIM steps: {args.num_inference_steps}")
    print(f"🎯 Guidance scale: {args.guidance_scale}")
    print(f"⚡ DDIM eta: {args.eta}")
    print(f"🎞️ Video length: {args.video_length}")
    print(f"🏃 Frame stride: {config['frame_stride']}")
    print("=" * 60)
    
    # Initialize pipeline
    print("🔧 Initializing DynamiCrafter Pipeline...")
    pipeline = DynamiCrafterImg2VideoPipeline(
        resolution=config['resolution_str'],
        device=args.device
    )
    
    # Process images
    if args.image_path:
        # Single image mode
        image_files = [args.image_path]
    else:
        # Batch mode - find all images in prompt directory
        image_files = find_image_files(args.prompt_dir)
    
    if not image_files:
        print("❌ No images found to process!")
        return
    
    print(f"📸 Found {len(image_files)} images to process")
    
    # Process each image
    total_start_time = time.time()
    
    for i, image_path in enumerate(image_files):
        print(f"\n🎬 Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            print(f"📐 Image size: {image.size}")
            
            # Load text prompt
            if args.prompt_text:
                prompt = args.prompt_text
            else:
                prompt = load_text_prompt(image_path)
            
            if not prompt:
                print("⚠️ No text prompt found, using default")
                prompt = "high quality video"
            
            print(f"📝 Prompt: {prompt}")
            
            # Generate video
            start_time = time.time()
            
            result = pipeline(
                image=image,
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                frame_stride=config['frame_stride'],
                num_frames=args.video_length,
                height=config['height'],
                width=config['width'],
                return_dict=True
            )
            
            generation_time = time.time() - start_time
            
            # Save video
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(args.output_dir, f"{image_name}.mp4")
            
            pipeline.save_video(result['videos'], output_path, fps=args.fps)
            
            # Check for NaN values
            video = result['videos']
            has_nan = False
            if hasattr(video, 'isnan'):
                has_nan = video.isnan().any()
            
            print(f"✅ Video generated successfully!")
            print(f"📊 Shape: {video.shape}")
            print(f"⏱️ Time: {generation_time:.2f}s")
            print(f"🔍 NaN check: {'❌ Found NaN' if has_nan else '✅ No NaN'}")
            print(f"💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start_time
    print(f"\n🎉 All processing completed!")
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"📂 Results saved in: {args.output_dir}")


if __name__ == '__main__':
    main() 