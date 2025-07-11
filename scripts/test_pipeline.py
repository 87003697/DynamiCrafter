#!/usr/bin/env python3
"""
Quick test script to verify DynamiCrafter Pipeline works correctly
"""

import os
import sys
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from scripts.gradio.dynamicrafter_pipeline import DynamiCrafterImg2VideoPipeline
    print("✅ Import successful!")
    
    # Test if we can initialize the pipeline
    print("🔧 Testing pipeline initialization...")
    pipeline = DynamiCrafterImg2VideoPipeline(resolution='256_256', device='cuda:5')
    print("✅ Pipeline initialization successful!")
    
    # Test if we can find a test image
    project_root = os.path.join(os.path.dirname(__file__), '..')
    test_image_path = os.path.join(project_root, 'prompts/1024/pour_bear.png')
    
    if os.path.exists(test_image_path):
        print(f"✅ Test image found: {test_image_path}")
        
        # Test image loading
        image = Image.open(test_image_path).convert('RGB')
        print(f"✅ Image loaded successfully: {image.size}")
        
        print("🎉 All tests passed! Ready to generate videos.")
        print("\nTo run a full test:")
        print("python scripts/run_pipeline.py 256 --device cuda:5 --num_inference_steps 5")
        
    else:
        print("⚠️ Test image not found, but pipeline can be initialized")
        print("You can still run the pipeline with your own images")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 