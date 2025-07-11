#!/usr/bin/env python3
"""
DynamiCrafter Pipeline Usage Examples
Demonstrates various ways to use the DynamiCrafterImg2VideoPipeline class
"""

import os
import sys
import time
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.gradio.dynamicrafter_pipeline import DynamiCrafterImg2VideoPipeline


def example_1_basic_usage():
    """Example 1: Basic usage with default parameters"""
    print("🎬 Example 1: Basic Usage")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DynamiCrafterImg2VideoPipeline(resolution='256_256')
    
    # Find a test image
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_image_path = os.path.join(project_root, 'prompts/1024/pour_bear.png')
    
    if not os.path.exists(test_image_path):
        print("❌ Test image not found, skipping example")
        return
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    
    # Generate video
    result = pipeline(
        image=image,
        prompt="a person walking in a beautiful garden",
        num_inference_steps=25,  # Fast generation
        guidance_scale=7.5,
        return_dict=True
    )
    
    # Save video
    output_path = "./results_examples/example1_basic.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.save_video(result['videos'], output_path)
    
    print(f"✅ Video saved: {output_path}")
    print(f"📊 Shape: {result['videos'].shape}")
    print()


def example_2_custom_parameters():
    """Example 2: Custom parameters and settings"""
    print("🎬 Example 2: Custom Parameters")
    print("=" * 50)
    
    # Initialize pipeline with custom device
    pipeline = DynamiCrafterImg2VideoPipeline(
        resolution='256_256',
        device='cuda:0'  # Specify GPU
    )
    
    # Find a test image
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_image_path = os.path.join(project_root, 'prompts/1024/pour_bear.png')
    
    if not os.path.exists(test_image_path):
        print("❌ Test image not found, skipping example")
        return
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    
    # Generate video with custom parameters
    result = pipeline(
        image=image,
        prompt="a majestic dragon flying through clouds",
        negative_prompt="blurry, low quality, distorted",
        num_inference_steps=30,
        guidance_scale=9.0,  # Higher guidance
        eta=0.5,  # Semi-stochastic
        frame_stride=2,  # Slower motion
        num_frames=16,
        height=256,
        width=256,
        return_dict=True
    )
    
    # Save video with custom FPS
    output_path = "./results_examples/example2_custom.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.save_video(result['videos'], output_path, fps=12)
    
    print(f"✅ Video saved: {output_path}")
    print(f"📊 Shape: {result['videos'].shape}")
    print()


def example_3_batch_processing():
    """Example 3: Batch processing multiple images"""
    print("🎬 Example 3: Batch Processing")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DynamiCrafterImg2VideoPipeline(resolution='256_256')
    
    # Find test images
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_images = [
        os.path.join(project_root, 'prompts/1024/pour_bear.png'),
        os.path.join(project_root, 'prompts/1024/robot01.png'),
        os.path.join(project_root, 'prompts/1024/astronaut04.png'),
    ]
    
    # Filter existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("❌ No test images found, skipping example")
        return
    
    # Process each image
    for i, image_path in enumerate(existing_images):
        print(f"📸 Processing image {i+1}/{len(existing_images)}: {os.path.basename(image_path)}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Generate video
        result = pipeline(
            image=image,
            prompt="dynamic movement with beautiful lighting",
            num_inference_steps=20,  # Fast generation for batch
            guidance_scale=7.5,
            return_dict=True
        )
        
        # Save video
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"./results_examples/example3_batch_{image_name}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pipeline.save_video(result['videos'], output_path)
        
        print(f"✅ Video saved: {output_path}")
    
    print()


def example_4_different_resolutions():
    """Example 4: Different resolutions"""
    print("🎬 Example 4: Different Resolutions")
    print("=" * 50)
    
    # Find a test image
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_image_path = os.path.join(project_root, 'prompts/1024/pour_bear.png')
    
    if not os.path.exists(test_image_path):
        print("❌ Test image not found, skipping example")
        return
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    
    # Test different resolutions
    resolutions = ['256_256', '512_512', '1024_1024']
    
    for resolution in resolutions:
        print(f"📐 Testing resolution: {resolution}")
        
        try:
            # Initialize pipeline for this resolution
            pipeline = DynamiCrafterImg2VideoPipeline(resolution=resolution)
            
            # Generate video
            start_time = time.time()
            result = pipeline(
                image=image,
                prompt="beautiful cinematic movement",
                num_inference_steps=15,  # Fast generation
                guidance_scale=7.5,
                return_dict=True
            )
            generation_time = time.time() - start_time
            
            # Save video
            output_path = f"./results_examples/example4_{resolution}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pipeline.save_video(result['videos'], output_path)
            
            print(f"✅ Video saved: {output_path}")
            print(f"📊 Shape: {result['videos'].shape}")
            print(f"⏱️ Time: {generation_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ Error with resolution {resolution}: {str(e)}")
            print()


def example_5_single_image_quick_test():
    """Example 5: Quick test with a single image"""
    print("🎬 Example 5: Quick Test")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DynamiCrafterImg2VideoPipeline(resolution='256_256')
    
    # Find a test image
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_image_path = os.path.join(project_root, 'prompts/1024/pour_bear.png')
    
    if not os.path.exists(test_image_path):
        print("❌ Test image not found, skipping example")
        return
    
    # Load image
    image = Image.open(test_image_path).convert('RGB')
    
    # Quick generation (minimal steps)
    start_time = time.time()
    result = pipeline(
        image=image,
        prompt="smooth motion",
        num_inference_steps=5,  # Very fast
        guidance_scale=7.5,
        eta=0.0,  # Deterministic
        return_dict=True
    )
    generation_time = time.time() - start_time
    
    # Save video
    output_path = "./results_examples/example5_quick.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline.save_video(result['videos'], output_path)
    
    print(f"✅ Video saved: {output_path}")
    print(f"📊 Shape: {result['videos'].shape}")
    print(f"⏱️ Time: {generation_time:.2f}s")
    print(f"🚀 Speed: {result['videos'].shape[2] / generation_time:.1f} fps generation")
    print()


def main():
    """Run all examples"""
    print("🎬 DynamiCrafter Pipeline Examples")
    print("=" * 60)
    print("This script demonstrates various usage patterns of the DynamiCrafterImg2VideoPipeline.")
    print()
    
    # Create output directory
    os.makedirs("./results_examples", exist_ok=True)
    
    # Run examples
    examples = [
        example_1_basic_usage,
        example_2_custom_parameters,
        example_3_batch_processing,
        example_4_different_resolutions,
        example_5_single_image_quick_test,
    ]
    
    for i, example_func in enumerate(examples, 1):
        print(f"Running Example {i}...")
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example {i} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
    
    print("🎉 All examples completed!")
    print("📂 Results saved in: ./results_examples/")


if __name__ == '__main__':
    main() 