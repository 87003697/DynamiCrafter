#!/bin/bash

# DynamiCrafter Guidance Pipeline - Simple Runner
# 注释掉 CUDA_VISIBLE_DEVICES 设置，直接使用 cuda:3
# export CUDA_VISIBLE_DEVICES=3

# Default settings
IMAGE="prompts/test/4.png"
PROMPT="A fat man is drinking beer"

# 🔧 FIX: 使用正确的分辨率设置
RESOLUTION="320_512"  # 修复：对于 512 模型使用正确的分辨率

# 🔧 FIX: 使用与标准inference一致的参数
FRAME_STRIDE=24       # 🔥 修复：512模型必须用24，不是3
GUIDANCE_SCALE=7.5    # 🔥 修复：使用标准值7.5
CFG_SCALE=7.5         # 🔥 修复：与guidance_scale一致
ETA=1.0               # 🔥 修复：使用标准值1.0，不是0.0

echo "Running DynamiCrafter Guidance Pipeline..."
echo "Image: $IMAGE"
echo "Prompt: $PROMPT"
echo "Resolution: $RESOLUTION"
echo "Frame Stride: $FRAME_STRIDE (512 model standard)"
echo "Guidance Scale: $GUIDANCE_SCALE"
echo "CFG Scale: $CFG_SCALE"
echo "ETA: $ETA"

# Basic run with enhanced saving and correct parameters
python generate_dynamicrafter_pipeline.py \
    --condition_image "$IMAGE" \
    --prompt "$PROMPT" \
    --resolution "$RESOLUTION" \
    --num_frames 16 \
    --frame_stride $FRAME_STRIDE \
    --guidance_scale $GUIDANCE_SCALE \
    --eta $ETA \
    --num_optimization_steps 1000 \
    --learning_rate 0.05 \
    --loss_type sds \
    --weight_type t \
    --cfg_scale $CFG_SCALE \
    --device cuda:3 \
    --save_results \
    --save_debug_images \
    --save_debug_videos \
    --save_process_video \
    --debug_save_interval 100 \
    --seed 42

echo "Done!"
