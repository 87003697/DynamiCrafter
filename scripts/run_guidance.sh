#!/bin/bash

# =============================================================================
# DynamiCrafter Guidance Pipeline Runner
# =============================================================================
# 
# 使用说明：
#   ./scripts/run_guidance.sh [选项]
#
# 选项：
#   test          - 运行测试
#   help          - 显示使用说明
#   run           - 运行完整生成（需要指定图像和提示词）
#   debug         - 调试模式运行
#
# 示例：
#   ./scripts/run_guidance.sh test
#   ./scripts/run_guidance.sh run "prompts/1024/pour_bear.png" "person walking in garden"
#   ./scripts/run_guidance.sh debug
#
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
GUIDANCE_SCRIPT="$PROJECT_ROOT/guidance_pipeline.py"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    # 检查 Python
    if ! command -v python &> /dev/null; then
        log_error "Python 未安装或未在 PATH 中"
        exit 1
    fi
    
    # 检查 CUDA（可选）
    if command -v nvidia-smi &> /dev/null; then
        log_success "检测到 CUDA 支持"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
    else
        log_warning "未检测到 CUDA，将使用 CPU 模式"
    fi
    
    # 检查 guidance_pipeline.py
    if [ ! -f "$GUIDANCE_SCRIPT" ]; then
        log_error "找不到 guidance_pipeline.py 文件: $GUIDANCE_SCRIPT"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 显示使用说明
show_usage() {
    log_header "DynamiCrafter Guidance Pipeline Runner"
    echo ""
    echo -e "${CYAN}使用方法：${NC}"
    echo "  $0 [选项]"
    echo ""
    echo -e "${CYAN}选项：${NC}"
    echo "  test                          - 运行测试"
    echo "  help                          - 显示使用说明"
    echo "  run [image_path] [prompt]     - 运行完整生成（参数可选，有默认值）"
    echo "  quick                         - 快速运行（使用默认参数，仅50步）"
    echo "  debug                         - 调试模式运行"
    echo "  info                          - 显示系统信息"
    echo ""
    echo -e "${CYAN}示例：${NC}"
    echo "  $0 test"
    echo "  $0 quick                                    # 快速测试，50步优化"
    echo "  $0 run                                      # 使用默认图像和提示词"
    echo "  $0 run \"image.jpg\"                         # 使用指定图像和默认提示词"
    echo "  $0 run \"image.jpg\" \"person walking\"       # 使用指定图像和提示词"
    echo "  $0 debug"
    echo "  $0 info"
    echo ""
    echo -e "${CYAN}默认值：${NC}"
    echo "  默认图像: 自动查找 prompts/ 目录下的测试图像"
    echo "  默认提示词: 'a person walking in a beautiful garden with flowers blooming'"
    echo ""
    echo -e "${CYAN}环境变量：${NC}"
    echo "  CUDA_VISIBLE_DEVICES          - 指定使用的 GPU"
    echo "  RESOLUTION                    - 视频分辨率 (256_256, 512_512, 1024_1024)"
    echo "  STEPS                         - 优化步数 (默认: 1000)"
    echo "  LOSS_TYPE                     - 损失类型 (sds, csd, rfds)"
    echo "  CFG_SCALE                     - CFG 比例 (默认: 7.5)"
    echo ""
    echo -e "${CYAN}示例使用环境变量：${NC}"
    echo "  RESOLUTION=512_512 STEPS=500 $0 run"
    echo "  CUDA_VISIBLE_DEVICES=0 LOSS_TYPE=sds $0 run \"image.jpg\" \"prompt\""
    echo ""
    echo -e "${CYAN}支持的测试图像位置：${NC}"
    echo "  - prompts/1024/pour_bear.png"
    echo "  - prompts/512_loop/24.png"
    echo "  - prompts/256/art.png"
    echo "  - prompts/256/bear.png"
    echo "  - prompts/1024/girl07.png"
    echo "  - prompts/512_loop/36.png"
}

# 运行测试
run_test() {
    log_header "运行 DynamiCrafter Guidance Pipeline 测试"
    
    check_dependencies
    
    log_info "开始测试..."
    log_info "脚本路径: $GUIDANCE_SCRIPT"
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 运行测试
    python "$GUIDANCE_SCRIPT" test
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "测试完成！"
        log_info "结果保存在: ./results_dynamicrafter_guidance/"
        log_info "调试视频保存在: ./debug_dynamicrafter_guidance/"
    else
        log_error "测试失败，退出码: $exit_code"
        exit $exit_code
    fi
}

# 运行完整生成
run_generation() {
    local image_path="$1"
    local prompt="$2"
    
    # === 新增：为参数提供默认值 ===
    # 如果没有提供图像路径，使用默认的测试图像
    if [ -z "$image_path" ]; then
        log_info "未提供图像路径，尝试使用默认测试图像..."
        
        # 按优先级查找测试图像
        local test_images=(
            "prompts/1024/pour_bear.png"
            "prompts/512_loop/24.png"
            "prompts/256/art.png"
            "prompts/256/bear.png"
            "prompts/1024/girl07.png"
            "prompts/512_loop/36.png"
        )
        
        for test_img in "${test_images[@]}"; do
            if [ -f "$test_img" ]; then
                image_path="$test_img"
                log_info "使用默认图像: $image_path"
                break
            fi
        done
        
        # 如果仍然没有找到图像，提示用户
        if [ -z "$image_path" ]; then
            log_error "未找到默认测试图像，请提供图像路径"
            echo "使用方法: $0 run <image_path> [prompt]"
            echo "或将测试图像放在以下位置之一："
            for test_img in "${test_images[@]}"; do
                echo "  - $test_img"
            done
            exit 1
        fi
    fi
    
    # 如果没有提供提示词，使用默认提示词
    if [ -z "$prompt" ]; then
        log_info "未提供提示词，使用默认提示词..."
        prompt="a person walking in a beautiful garden with flowers blooming"
        log_info "使用默认提示词: $prompt"
    fi
    
    log_header "运行 DynamiCrafter Guidance Pipeline 生成"
    
    check_dependencies
    
    # 检查图像文件
    if [ ! -f "$image_path" ]; then
        log_error "图像文件不存在: $image_path"
        exit 1
    fi
    
    log_info "图像路径: $image_path"
    log_info "提示词: $prompt"
    
    # 环境变量设置
    local resolution="${RESOLUTION:-256_256}"
    local steps="${STEPS:-1000}"
    local loss_type="${LOSS_TYPE:-sds}"
    local cfg_scale="${CFG_SCALE:-7.5}"
    
    log_info "分辨率: $resolution"
    log_info "优化步数: $steps"
    log_info "损失类型: $loss_type"
    log_info "CFG 比例: $cfg_scale"
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 创建 Python 脚本运行生成
    cat > run_temp.py << EOF
from guidance_pipeline import DynamiCrafterGuidancePipeline
from PIL import Image
import os

# 初始化 pipeline
pipeline = DynamiCrafterGuidancePipeline(resolution='$resolution')

# 加载图像
image = Image.open('$image_path')

# 运行生成
result = pipeline(
    image=image,
    prompt='$prompt',
    num_optimization_steps=$steps,
    loss_type='$loss_type',
    cfg_scale=$cfg_scale,
    save_debug_videos=True,
    debug_save_interval=100,
    debug_save_path='./debug_dynamicrafter_guidance',
    return_dict=True
)

# 保存视频
output_dir = './results_dynamicrafter_guidance/'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'generated_video.mp4')

pipeline.save_video(result['videos'], output_path)
print(f"✅ 视频生成完成: {output_path}")
EOF

    # 运行生成
    python run_temp.py
    
    local exit_code=$?
    
    # 清理临时文件
    rm -f run_temp.py
    
    if [ $exit_code -eq 0 ]; then
        log_success "生成完成！"
        log_info "结果保存在: ./results_dynamicrafter_guidance/"
        log_info "调试视频保存在: ./debug_dynamicrafter_guidance/"
    else
        log_error "生成失败，退出码: $exit_code"
        exit $exit_code
    fi
}

# 调试模式运行
run_debug() {
    log_header "调试模式运行 DynamiCrafter Guidance Pipeline"
    
    check_dependencies
    
    log_info "开始调试模式..."
    
    # 设置环境变量
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export CUDA_LAUNCH_BLOCKING=1  # 调试 CUDA 错误
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    
    # 运行 Python 调试器
    python -m pdb "$GUIDANCE_SCRIPT" test
}

# 显示系统信息
show_system_info() {
    log_header "系统信息"
    
    echo -e "${CYAN}Python 版本：${NC}"
    python --version
    
    echo -e "${CYAN}工作目录：${NC}"
    pwd
    
    echo -e "${CYAN}项目根目录：${NC}"
    echo "$PROJECT_ROOT"
    
    echo -e "${CYAN}GPU 信息：${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader
    else
        echo "未检测到 NVIDIA GPU"
    fi
    
    echo -e "${CYAN}环境变量：${NC}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-未设置}"
    echo "RESOLUTION: ${RESOLUTION:-256_256 (默认)}"
    echo "STEPS: ${STEPS:-1000 (默认)}"
    echo "LOSS_TYPE: ${LOSS_TYPE:-sds (默认)}"
    echo "CFG_SCALE: ${CFG_SCALE:-7.5 (默认)}"
}

# 主函数
main() {
    case "${1:-help}" in
        "test")
            run_test
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        "run")
            run_generation "$2" "$3"
            ;;
        "quick")
            # 快速运行：使用默认参数的简化版本
            log_header "快速运行 DynamiCrafter Guidance Pipeline"
            log_info "使用默认参数进行快速测试..."
            STEPS=50 run_generation
            ;;
        "debug")
            run_debug
            ;;
        "info")
            show_system_info
            ;;
        *)
            log_info "显示使用说明..."
            python "$GUIDANCE_SCRIPT"
            echo ""
            show_usage
            ;;
    esac
}

# 运行主函数
main "$@"
