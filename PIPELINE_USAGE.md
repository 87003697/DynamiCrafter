# DynamiCrafter Pipeline Usage Guide

本指南说明如何使用 `DynamiCrafterImg2VideoPipeline` 类和相关脚本来生成视频。

## 📁 文件概览

```
scripts/
├── run_pipeline.py          # Python脚本，使用pipeline生成视频
├── run_pipeline.sh          # Shell脚本，类似于原始run.sh
├── pipeline_examples.py     # 使用示例脚本
└── gradio/
    └── dynamicrafter_pipeline.py  # Pipeline类定义
```

## 🚀 快速开始

### 1. 使用Shell脚本（推荐）

```bash
# 基本用法
./scripts/run_pipeline.sh 256

# 指定随机种子
./scripts/run_pipeline.sh 256 123

# 指定GPU设备
./scripts/run_pipeline.sh 256 123 cuda:1

# 不同分辨率
./scripts/run_pipeline.sh 512    # 512x320
./scripts/run_pipeline.sh 1024   # 1024x576
```

### 2. 使用Python脚本

```bash
# 基本用法
python scripts/run_pipeline.py 256

# 自定义参数
python scripts/run_pipeline.py 256 \
    --seed 123 \
    --device cuda:0 \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --eta 1.0 \
    --video_length 16

# 单张图片生成
python scripts/run_pipeline.py 256 \
    --image_path prompts/1024/pour_bear.png \
    --prompt_text "a person walking in a beautiful garden" \
    --output_dir results/single_test

# 自定义输出目录
python scripts/run_pipeline.py 256 \
    --prompt_dir prompts/256/ \
    --output_dir results/my_videos
```

### 3. 运行示例脚本

```bash
# 运行所有示例
python scripts/pipeline_examples.py

# 查看不同的使用模式
```

## 📖 详细用法

### Python API 使用

```python
from scripts.gradio.dynamicrafter_pipeline import DynamiCrafterImg2VideoPipeline
from PIL import Image

# 1. 初始化pipeline
pipeline = DynamiCrafterImg2VideoPipeline(
    resolution='256_256',  # 或 '512_512', '1024_1024'
    device='cuda:0'        # 或 'cpu'
)

# 2. 加载图片
image = Image.open('your_image.jpg').convert('RGB')

# 3. 生成视频
result = pipeline(
    image=image,
    prompt="your text prompt here",
    negative_prompt="blurry, low quality",  # 可选
    num_inference_steps=50,                 # DDIM步数
    guidance_scale=7.5,                     # 引导强度
    eta=1.0,                               # DDIM eta参数
    frame_stride=3,                        # 帧步长
    num_frames=16,                         # 帧数
    height=256,                            # 高度
    width=256,                             # 宽度
    return_dict=True
)

# 4. 保存视频
pipeline.save_video(result['videos'], 'output.mp4', fps=8)
```

### Shell脚本参数说明

```bash
./scripts/run_pipeline.sh <resolution> [seed] [device]
```

- `resolution`: 必需，256/512/1024
- `seed`: 可选，随机种子（默认：123）
- `device`: 可选，GPU设备（默认：cuda:0）

### Python脚本参数说明

```bash
python scripts/run_pipeline.py <resolution> [options]
```

**主要参数：**
- `resolution`: 必需，256/512/1024
- `--seed`: 随机种子（默认：123）
- `--device`: GPU设备（默认：cuda:0）

**生成参数：**
- `--num_inference_steps`: DDIM步数（默认：50）
- `--guidance_scale`: 引导强度（默认：7.5）
- `--eta`: DDIM eta参数（默认：1.0）
- `--video_length`: 视频帧数（默认：16）

**输入输出：**
- `--prompt_dir`: 输入图片目录（默认：prompts/{resolution}/）
- `--output_dir`: 输出目录（默认：results/dynamicrafter_pipeline_{resolution}_seed{seed}/）
- `--image_path`: 单张图片路径
- `--prompt_text`: 文本提示词

**其他选项：**
- `--batch_size`: 批次大小（默认：1）
- `--fps`: 输出视频帧率（默认：8）

## 🎯 不同分辨率配置

| 分辨率 | 输出尺寸 | Frame Stride | 推荐用途 |
|--------|----------|--------------|----------|
| 256    | 256x256  | 3            | 快速预览、测试 |
| 512    | 512x320  | 24           | 平衡质量和速度 |
| 1024   | 1024x576 | 10           | 高质量输出 |

## 📂 输入格式

### 图片格式
支持的图片格式：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

### 提示词格式
- 直接在命令行指定：`--prompt_text "your prompt"`
- 使用.txt文件：与图片同名的.txt文件（如：`image.png` → `image.txt`）
- 目录批处理：每张图片对应一个.txt文件

### 目录结构示例
```
prompts/256/
├── image1.png
├── image1.txt          # 包含提示词
├── image2.jpg
├── image2.txt
└── ...
```

## 🔧 性能优化

### 快速生成（测试用）
```bash
python scripts/run_pipeline.py 256 \
    --num_inference_steps 10 \
    --guidance_scale 5.0
```

### 高质量生成
```bash
python scripts/run_pipeline.py 1024 \
    --num_inference_steps 50 \
    --guidance_scale 9.0 \
    --eta 0.0
```

### 批量处理
```bash
# 处理整个目录
python scripts/run_pipeline.py 256 \
    --prompt_dir prompts/256/ \
    --num_inference_steps 25
```

## 📊 输出格式

### 视频文件
- 格式：MP4
- 默认帧率：8 FPS
- 编码：H.264

### 命名规则
- 批处理：`{image_name}.mp4`
- 输出目录：`results/dynamicrafter_pipeline_{resolution}_seed{seed}/`

## 🐛 常见问题

### 1. 内存不足
```bash
# 使用较小分辨率
python scripts/run_pipeline.py 256

# 减少推理步数
python scripts/run_pipeline.py 256 --num_inference_steps 20
```

### 2. GPU设备选择
```bash
# 检查可用GPU
nvidia-smi

# 指定GPU
python scripts/run_pipeline.py 256 --device cuda:1
```

### 3. 模型下载
模型会自动从 Hugging Face 下载到 `checkpoints/` 目录。

### 4. NaN问题
Pipeline已集成修复方案，自动处理NaN问题。

## 📝 示例命令

```bash
# 1. 快速测试
./scripts/run_pipeline.sh 256

# 2. 高质量生成
./scripts/run_pipeline.sh 1024 456 cuda:0

# 3. 自定义参数
python scripts/run_pipeline.py 512 \
    --seed 789 \
    --num_inference_steps 30 \
    --guidance_scale 8.0 \
    --prompt_text "a beautiful landscape with flowing water"

# 4. 批量处理
python scripts/run_pipeline.py 256 \
    --prompt_dir my_images/ \
    --output_dir my_results/ \
    --num_inference_steps 25

# 5. 单张图片
python scripts/run_pipeline.py 256 \
    --image_path test.jpg \
    --prompt_text "dynamic movement" \
    --output_dir single_output/
```

## 🎬 与原始run.sh的对比

| 功能 | 原始run.sh | run_pipeline.sh |
|------|------------|-----------------|
| 接口 | inference.py | DynamiCrafterImg2VideoPipeline |
| NaN修复 | 需要额外设置 | 自动集成 |
| 参数控制 | 有限 | 灵活 |
| 错误处理 | 基本 | 完善 |
| 批处理 | 单次 | 支持 |
| 设备选择 | 环境变量 | 参数指定 |

## 🔗 相关文件

- `scripts/gradio/dynamicrafter_pipeline.py`: Pipeline类实现
- `scripts/evaluation/inference.py`: 原始推理脚本
- `scripts/run.sh`: 原始运行脚本 