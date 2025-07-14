#!/bin/bash

# DynamiCrafter Guidance Pipeline - Simple Runner
# 注释掉 CUDA_VISIBLE_DEVICES 设置，直接使用 cuda:3
# export CUDA_VISIBLE_DEVICES=3

# Default settings
IMAGE="prompts/256/bear.png"
PROMPT="a brown bear is walking in a zoo enclosure, some rocks around"

echo "Running DynamiCrafter Guidance Pipeline..."
echo "Image: $IMAGE"
echo "Prompt: $PROMPT"

# Basic run with enhanced saving
python generate_dynamicrafter_pipeline.py \
    --condition_image "$IMAGE" \
    --prompt "$PROMPT" \
    --resolution "256_256" \
    --num_frames 16 \
    --num_optimization_steps 1000 \
    --learning_rate 0.05 \
    --loss_type csd \
    --weight_type ada \
    --cfg_scale 7.5 \
    --device cuda:3 \
    --save_results \
    --save_debug_images \
    --save_debug_videos \
    --save_process_video \
    --debug_save_interval 100 \
    --seed 42

echo "Done!"
