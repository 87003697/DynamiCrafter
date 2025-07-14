#!/bin/bash

# DynamiCrafter Guidance Pipeline Runner
# Enhanced version with comprehensive result saving
# Updated: 2024-01-XX

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Default parameters
DEFAULT_IMAGE_DIR="prompts"
DEFAULT_PROMPT_FILE="prompts/prompt.txt"
DEFAULT_PROMPT="A beautiful landscape with flowing water"
DEFAULT_STEPS=50
DEFAULT_LR=0.01
DEFAULT_LOSS_TYPE="sds"
DEFAULT_CFG_SCALE=7.5
DEFAULT_RESULTS_DIR="results_dynamicrafter_guidance"
DEFAULT_DEBUG_INTERVAL=10

# Enhanced saving parameters
DEFAULT_SAVE_RESULTS=true
DEFAULT_SAVE_DEBUG_IMAGES=true
DEFAULT_SAVE_DEBUG_VIDEOS=true
DEFAULT_SAVE_PROCESS_VIDEO=true

# Print header
print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                         DynamiCrafter Guidance Pipeline Runner                         ║${NC}"
    echo -e "${CYAN}║                              Enhanced Saving Version                                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Print help
print_help() {
    echo -e "${WHITE}Usage: $0 [MODE] [OPTIONS]${NC}"
    echo ""
    echo -e "${YELLOW}MODES:${NC}"
    echo -e "  ${GREEN}test${NC}     - Quick test run (10 steps, basic saving)"
    echo -e "  ${GREEN}run${NC}      - Full optimization run (50 steps, enhanced saving)"
    echo -e "  ${GREEN}debug${NC}    - Debug mode with frequent saves (20 steps, interval=5)"
    echo -e "  ${GREEN}quick${NC}    - Quick generation (25 steps, standard saving)"
    echo -e "  ${GREEN}info${NC}     - Show system information"
    echo -e "  ${GREEN}help${NC}     - Show this help message"
    echo ""
    echo -e "${YELLOW}OPTIONS:${NC}"
    echo -e "  ${BLUE}--image PATH${NC}            Input image path (auto-detects from prompts/ if not specified)"
    echo -e "  ${BLUE}--prompt TEXT${NC}           Text prompt (reads from prompts/prompt.txt if not specified)"  
    echo -e "  ${BLUE}--steps N${NC}               Number of optimization steps (default: $DEFAULT_STEPS)"
    echo -e "  ${BLUE}--lr FLOAT${NC}              Learning rate (default: $DEFAULT_LR)"
    echo -e "  ${BLUE}--loss TYPE${NC}             Loss type: sds, csd, rfds (default: $DEFAULT_LOSS_TYPE)"
    echo -e "  ${BLUE}--cfg_scale FLOAT${NC}       CFG scale (default: $DEFAULT_CFG_SCALE)"
    echo -e "  ${BLUE}--results_dir PATH${NC}      Results directory (default: $DEFAULT_RESULTS_DIR)"
    echo -e "  ${BLUE}--debug_interval N${NC}      Debug save interval (default: $DEFAULT_DEBUG_INTERVAL)"
    echo ""
    echo -e "${YELLOW}ENHANCED SAVING OPTIONS:${NC}"
    echo -e "  ${BLUE}--save_results BOOL${NC}     Enable organized result saving (default: $DEFAULT_SAVE_RESULTS)"
    echo -e "  ${BLUE}--save_debug_images BOOL${NC} Save debug images (default: $DEFAULT_SAVE_DEBUG_IMAGES)"
    echo -e "  ${BLUE}--save_debug_videos BOOL${NC} Save debug videos (default: $DEFAULT_SAVE_DEBUG_VIDEOS)"
    echo -e "  ${BLUE}--save_process_video BOOL${NC} Create optimization process video (default: $DEFAULT_SAVE_PROCESS_VIDEO)"
    echo ""
    echo -e "${YELLOW}EXAMPLES:${NC}"
    echo -e "  ${GREEN}$0 test${NC}                                    # Quick test"
    echo -e "  ${GREEN}$0 run --steps 100 --lr 0.05${NC}              # Full run with custom parameters"
    echo -e "  ${GREEN}$0 debug --image my_image.jpg${NC}              # Debug mode with specific image"
    echo -e "  ${GREEN}$0 quick --prompt \"A serene mountain lake\"${NC}  # Quick run with custom prompt"
    echo ""
}

# Auto-detect image file
auto_detect_image() {
    local image_dir="$1"
    
    if [ -d "$image_dir" ]; then
        # First, look for common image extensions in the root directory
        for ext in jpg jpeg png bmp tiff webp; do
            local found_file=$(find "$image_dir" -maxdepth 1 -iname "*.$ext" -type f | head -1)
            if [ -n "$found_file" ]; then
                echo "$found_file"
                return 0
            fi
        done
        
        # If not found in root, search in subdirectories
        for ext in jpg jpeg png bmp tiff webp; do
            local found_file=$(find "$image_dir" -maxdepth 2 -iname "*.$ext" -type f | head -1)
            if [ -n "$found_file" ]; then
                echo "$found_file"
                return 0
            fi
        done
    fi
    
    return 1
}

# Auto-detect prompt
auto_detect_prompt() {
    local prompt_file="$1"
    
    # Try to read from prompt file
    if [ -f "$prompt_file" ]; then
        local prompt=$(cat "$prompt_file" | head -1 | xargs)
        if [ -n "$prompt" ]; then
            echo "$prompt"
            return 0
        fi
    fi
    
    # Try to read from test_prompts.txt in subdirectories
    for subdir in prompts/*/; do
        if [ -f "${subdir}test_prompts.txt" ]; then
            local prompt=$(cat "${subdir}test_prompts.txt" | head -1 | xargs)
            if [ -n "$prompt" ]; then
                echo "$prompt"
                return 0
            fi
        fi
    done
    
    # Use default prompt if nothing found
    echo "$DEFAULT_PROMPT"
    return 0
}

# Show system information
show_system_info() {
    echo -e "${CYAN}System Information:${NC}"
    echo -e "  ${BLUE}Python:${NC} $(python --version 2>&1)"
    echo -e "  ${BLUE}PyTorch:${NC} $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo 'Not installed')"
    echo -e "  ${BLUE}CUDA Available:${NC} $(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo 'Unknown')"
    echo -e "  ${BLUE}GPU Count:${NC} $(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 'Unknown')"
    echo -e "  ${BLUE}Current GPU:${NC} $(nvidia-smi -L 2>/dev/null | head -1 || echo 'No GPU info')"
    echo -e "  ${BLUE}Working Directory:${NC} $(pwd)"
    echo -e "  ${BLUE}DynamiCrafter:${NC} $([ -f "guidance_pipeline.py" ] && echo 'Available' || echo 'Not found')"
    echo ""
}

# Run the pipeline
run_pipeline() {
    local mode="$1"
    shift
    
    # Parse arguments
    local image_path=""
    local prompt=""
    local steps="$DEFAULT_STEPS"
    local lr="$DEFAULT_LR"
    local loss_type="$DEFAULT_LOSS_TYPE"
    local cfg_scale="$DEFAULT_CFG_SCALE"
    local results_dir="$DEFAULT_RESULTS_DIR"
    local debug_interval="$DEFAULT_DEBUG_INTERVAL"
    
    # Enhanced saving parameters
    local save_results="$DEFAULT_SAVE_RESULTS"
    local save_debug_images="$DEFAULT_SAVE_DEBUG_IMAGES"
    local save_debug_videos="$DEFAULT_SAVE_DEBUG_VIDEOS"
    local save_process_video="$DEFAULT_SAVE_PROCESS_VIDEO"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --image)
                image_path="$2"
                shift 2
                ;;
            --prompt)
                prompt="$2"
                shift 2
                ;;
            --steps)
                steps="$2"
                shift 2
                ;;
            --lr)
                lr="$2"
                shift 2
                ;;
            --loss)
                loss_type="$2"
                shift 2
                ;;
            --cfg_scale)
                cfg_scale="$2"
                shift 2
                ;;
            --results_dir)
                results_dir="$2"
                shift 2
                ;;
            --debug_interval)
                debug_interval="$2"
                shift 2
                ;;
            --save_results)
                save_results="$2"
                shift 2
                ;;
            --save_debug_images)
                save_debug_images="$2"
                shift 2
                ;;
            --save_debug_videos)
                save_debug_videos="$2"
                shift 2
                ;;
            --save_process_video)
                save_process_video="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                exit 1
                ;;
        esac
    done
    
    # Set mode-specific parameters
    case "$mode" in
        "test")
            steps=10
            debug_interval=5
            save_debug_images=false
            save_debug_videos=true
            save_process_video=false
            ;;
        "run")
            steps=50
            debug_interval=10
            save_debug_images=true
            save_debug_videos=true
            save_process_video=true
            ;;
        "debug")
            steps=20
            debug_interval=5
            save_debug_images=true
            save_debug_videos=true
            save_process_video=true
            ;;
        "quick")
            steps=25
            debug_interval=10
            save_debug_images=false
            save_debug_videos=true
            save_process_video=false
            ;;
    esac
    
    # Auto-detect image if not specified
    if [ -z "$image_path" ]; then
        echo -e "${YELLOW}Auto-detecting image...${NC}"
        image_path=$(auto_detect_image "$DEFAULT_IMAGE_DIR")
        if [ -z "$image_path" ]; then
            echo -e "${RED}Error: No image found in $DEFAULT_IMAGE_DIR${NC}"
            echo -e "${WHITE}Please specify an image with --image PATH${NC}"
            exit 1
        fi
        echo -e "${GREEN}Found image: $image_path${NC}"
    fi
    
    # Auto-detect prompt if not specified
    if [ -z "$prompt" ]; then
        echo -e "${YELLOW}Auto-detecting prompt...${NC}"
        prompt=$(auto_detect_prompt "$DEFAULT_PROMPT_FILE")
        echo -e "${GREEN}Using prompt: $prompt${NC}"
    fi
    
    # Print configuration
    echo -e "${CYAN}Configuration:${NC}"
    echo -e "  ${BLUE}Mode:${NC} $mode"
    echo -e "  ${BLUE}Image:${NC} $image_path"
    echo -e "  ${BLUE}Prompt:${NC} $prompt"
    echo -e "  ${BLUE}Steps:${NC} $steps"
    echo -e "  ${BLUE}Learning Rate:${NC} $lr"
    echo -e "  ${BLUE}Loss Type:${NC} $loss_type"
    echo -e "  ${BLUE}CFG Scale:${NC} $cfg_scale"
    echo -e "  ${BLUE}Results Directory:${NC} $results_dir"
    echo -e "  ${BLUE}Debug Interval:${NC} $debug_interval"
    echo -e "${CYAN}Enhanced Saving:${NC}"
    echo -e "  ${BLUE}Save Results:${NC} $save_results"
    echo -e "  ${BLUE}Save Debug Images:${NC} $save_debug_images"
    echo -e "  ${BLUE}Save Debug Videos:${NC} $save_debug_videos"
    echo -e "  ${BLUE}Save Process Video:${NC} $save_process_video"
    echo ""
    
    # Create results directory
    mkdir -p "$results_dir"
    
    # Run the pipeline
    echo -e "${GREEN}Starting DynamiCrafter Guidance Pipeline...${NC}"
    echo -e "${WHITE}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
    export PROMPT_TEXT="$prompt"
    
    # Convert shell boolean to Python boolean
    local py_save_results=$([ "$save_results" = "true" ] && echo "True" || echo "False")
    local py_save_debug_images=$([ "$save_debug_images" = "true" ] && echo "True" || echo "False")
    local py_save_debug_videos=$([ "$save_debug_videos" = "true" ] && echo "True" || echo "False")
    local py_save_process_video=$([ "$save_process_video" = "true" ] && echo "True" || echo "False")
    
    # Build Python command
    local python_cmd="python -c \"
import sys
import os
sys.path.insert(0, '.')
from guidance_pipeline import DynamiCrafterGuidancePipeline
from PIL import Image
import torch

# Initialize pipeline
pipeline = DynamiCrafterGuidancePipeline()

# Load image
image = Image.open('$image_path').convert('RGB')

# Get prompt from environment variable
prompt = os.environ.get('PROMPT_TEXT', '$DEFAULT_PROMPT')

# Run optimization
result = pipeline(
    image=image,
    prompt=prompt,
    num_optimization_steps=$steps,
    learning_rate=$lr,
    loss_type='$loss_type',
    cfg_scale=$cfg_scale,
    save_results=$py_save_results,
    results_dir='$results_dir',
    save_debug_images=$py_save_debug_images,
    save_debug_videos=$py_save_debug_videos,
    save_process_video=$py_save_process_video,
    debug_save_interval=$debug_interval,
    output_type='tensor'
)

print('\\n🎉 Pipeline completed successfully!')
print(f'📁 Results saved to: $results_dir')
\""
    
    # Execute the command
    eval "$python_cmd"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ Pipeline completed successfully!${NC}"
        echo -e "${CYAN}📁 Results saved to: $results_dir${NC}"
        
        # Show output structure
        if [ "$save_results" = "true" ]; then
            echo -e "${CYAN}📋 Output structure:${NC}"
            find "$results_dir" -type d -name "*$(date +%Y%m%d)*" -exec ls -la {} \; 2>/dev/null | head -20
        fi
    else
        echo -e "${RED}❌ Pipeline failed with exit code: $exit_code${NC}"
        exit $exit_code
    fi
}

# Main script logic
main() {
    print_header
    
    # Handle no arguments
    if [ $# -eq 0 ]; then
        echo -e "${YELLOW}No arguments provided. Use '$0 help' for usage information.${NC}"
        echo -e "${GREEN}Running in 'quick' mode by default...${NC}"
        echo ""
        run_pipeline "quick"
        return
    fi
    
    # Handle mode
    local mode="$1"
    shift
    
    case "$mode" in
        "test"|"run"|"debug"|"quick")
            run_pipeline "$mode" "$@"
            ;;
        "info")
            show_system_info
            ;;
        "help"|"-h"|"--help")
            print_help
            ;;
        *)
            echo -e "${RED}Unknown mode: $mode${NC}"
            echo -e "${WHITE}Use '$0 help' for available modes.${NC}"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
