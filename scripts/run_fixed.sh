#!/bin/bash
# DynamiCrafter inference script with minimal fixed scheduler

version=$1
seed=123

# Validate input
if [[ "$1" != "256" && "$1" != "512" && "$1" != "1024" ]]; then
    echo "Usage: $0 [256|512|1024]"
    exit 1
fi

# Set resolution-specific parameters
case $1 in
    256) H=256; FS=3 ;;
    512) H=320; FS=24 ;;
    1024) H=576; FS=10 ;;
esac

echo "🚀 Running DynamiCrafter inference with minimal fixed scheduler"
echo "📊 Resolution: ${1}x${1}, Height: ${H}, FS: ${FS}"

# Build command
cmd="CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path checkpoints/dynamicrafter_${1}_v1/model.ckpt \
--config configs/inference_${1}_v1.0.yaml \
--savedir results/dynamicrafter_${1}_fixed_seed${seed} \
--n_samples 1 --bs 1 --height ${H} --width ${1} \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 --ddim_eta 1.0 \
--prompt_dir prompts/${1}/ \
--text_input --video_length 16 --frame_stride ${FS} \
--use_fixed_scheduler"

# Add extra parameters for 512/1024
if [[ "$1" != "256" ]]; then
    cmd+=" --timestep_spacing uniform_trailing --guidance_rescale 0.7 --perframe_ae"
fi

# Execute command
echo "🔧 Executing: $(basename $0) $1"
eval $cmd

echo "✅ Inference completed! Results in: results/dynamicrafter_${1}_fixed_seed${seed}" 