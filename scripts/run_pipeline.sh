#!/bin/bash

# DynamiCrafter Pipeline Runner Script
# Similar to run.sh but uses the DynamiCrafterImg2VideoPipeline class

version=$1  # 256, 512, 1024
seed=${2:-123}  # Default seed is 123
device=${3:-"cuda:5"}  # Default device

# Validate input
if [ -z "$version" ] || [[ ! "$version" =~ ^(256|512|1024)$ ]]; then
    echo "Usage: $0 <version> [seed] [device]"
    echo "  version: 256, 512, or 1024"
    echo "  seed: random seed (default: 123)"
    echo "  device: cuda device (default: cuda:0)"
    echo ""
    echo "Examples:"
    echo "  $0 256"
    echo "  $0 512 123 cuda:1"
    echo "  $0 1024 456 cuda:0"
    exit 1
fi

# Set parameters based on version
name="dynamicrafter_pipeline_${version}_seed${seed}"
prompt_dir="prompts/${version}/"
res_dir="results"

# Common parameters
num_inference_steps=50
guidance_scale=7.5
ddim_eta=1.0
video_length=16
fps=8

# Version-specific parameters
if [ "$version" == "256" ]; then
    echo "🎬 Running DynamiCrafter Pipeline for 256x256 resolution"
    python3 scripts/run_pipeline.py $version \
        --seed $seed \
        --device $device \
        --num_inference_steps $num_inference_steps \
        --guidance_scale $guidance_scale \
        --eta $ddim_eta \
        --video_length $video_length \
        --prompt_dir $prompt_dir \
        --output_dir $res_dir/$name \
        --fps $fps
elif [ "$version" == "512" ]; then
    echo "🎬 Running DynamiCrafter Pipeline for 512x320 resolution"
    python3 scripts/run_pipeline.py $version \
        --seed $seed \
        --device $device \
        --num_inference_steps $num_inference_steps \
        --guidance_scale $guidance_scale \
        --eta $ddim_eta \
        --video_length $video_length \
        --prompt_dir $prompt_dir \
        --output_dir $res_dir/$name \
        --fps $fps
elif [ "$version" == "1024" ]; then
    echo "🎬 Running DynamiCrafter Pipeline for 1024x576 resolution"
    python3 scripts/run_pipeline.py $version \
        --seed $seed \
        --device $device \
        --num_inference_steps $num_inference_steps \
        --guidance_scale $guidance_scale \
        --eta $ddim_eta \
        --video_length $video_length \
        --prompt_dir $prompt_dir \
        --output_dir $res_dir/$name \
        --fps $fps
fi

echo "🎉 Pipeline execution completed!"
echo "📂 Results saved in: $res_dir/$name" 