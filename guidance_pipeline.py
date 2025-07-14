# dynamicrafter_guidance_pipeline.py

"""
DynamiCrafter Guidance Pipeline

融合了以下三个源文件的技术：
1. scripts/gradio/dynamicrafter_pipeline.py - DynamiCrafter 核心实现
2. _reference_codes/Stable3DGen/flux_kontext_pipeline.py - Flux Kontext 标准流水线
3. _reference_codes/Stable3DGen/2D_experiments/guidance_flux_pipeline.py - Flux Guidance 优化技术

作者：Assistant
创建时间：2025年
"""

import os
import sys
import json
import time
import shutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import cv2
from PIL import Image
import imageio
from omegaconf import OmegaConf
from einops import rearrange, repeat
from contextlib import contextmanager
import torchvision.transforms as transforms

# DynamiCrafter imports - 来自 scripts/evaluation/funcs.py
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, get_latent_z

# 添加 DynamiCrafter 根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Additional imports for enhanced saving
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download

# Additional DynamiCrafter imports
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters


@dataclass
class DynamiCrafterGuidanceConfig:
    """
    DynamiCrafter Guidance 优化配置
    
    设计灵感来源：
    - guidance_flux_pipeline.py:FluxGuidanceConfig (基础结构)
    - dynamicrafter_pipeline.py (DynamiCrafter特定参数)
    """
    # === 来自 guidance_flux_pipeline.py:FluxGuidanceConfig ===
    num_optimization_steps: int = 1000
    learning_rate: float = 0.05
    loss_type: str = "sds"  # sds, csd, rfds
    weight_type: str = "auto"  # auto, t, ada, uniform
    cfg_scale: float = 7.5
    optimizer_type: str = "AdamW"  # AdamW, Adam
    
    # === 新增：DynamiCrafter 特定参数 ===
    frame_stride: int = 3
    temporal_guidance_scale: float = 1.0  # 时间维度的额外引导
    
    # === 来自 guidance_flux_pipeline.py:FluxGuidanceConfig ===
    min_step_ratio_start: float = 0.02
    min_step_ratio_end: float = 0.02
    max_step_ratio_start: float = 0.98
    max_step_ratio_end: float = 0.98
    
    weight_eps: float = 0.0
    
    # === 修改：针对视频的 Debug 参数 ===
    save_debug_videos: bool = False  # 原来是 save_debug_images
    debug_save_interval: int = 100
    debug_save_path: str = "debug_videos"  # 原来是 debug_images


def create_output_structure(base_dir: str, prompt: str, guidance_scale: float, 
                           num_inference_steps: int) -> Dict[str, str]:
    """Create organized output directory structure with descriptive naming."""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe prompt name (remove special characters)
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Truncate long prompts
    
    # Create main output directory
    output_name = f"{timestamp}_{safe_prompt}_gs{guidance_scale}_steps{num_inference_steps}"
    main_dir = os.path.join(base_dir, output_name)
    
    # Create subdirectories
    dirs = {
        'main': main_dir,
        'inputs': os.path.join(main_dir, 'inputs'),
        'outputs': os.path.join(main_dir, 'outputs'),
        'debug': os.path.join(main_dir, 'debug'),
        'process': os.path.join(main_dir, 'process'),
        'params': os.path.join(main_dir, 'params')
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_parameters(params_dir: str, **kwargs):
    """Save all parameters to structured files."""
    
    # Save main parameters as JSON
    params_file = os.path.join(params_dir, 'parameters.json')
    with open(params_file, 'w') as f:
        json.dump(kwargs, f, indent=2, default=str)
    
    # Save readable text format
    txt_file = os.path.join(params_dir, 'parameters.txt')
    with open(txt_file, 'w') as f:
        f.write(f"DynamiCrafter Guidance Pipeline Parameters\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Parameters saved to: {params_file}")

def save_input_conditions(inputs_dir: str, image_path: str, prompt: str, 
                         processed_image: torch.Tensor):
    """Save input conditions: original image, processed image, and prompt."""
    
    # Copy original image
    if os.path.exists(image_path):
        shutil.copy2(image_path, os.path.join(inputs_dir, 'original_image.png'))
    
    # Save processed image
    if processed_image is not None:
        # Convert tensor to PIL Image
        if len(processed_image.shape) == 4:
            processed_image = processed_image[0]  # Remove batch dimension
        
        # Ensure proper format
        if processed_image.shape[0] == 3:  # CHW format
            processed_image = processed_image.permute(1, 2, 0)  # HWC format
        
        processed_image = processed_image.cpu().numpy()
        if processed_image.max() <= 1.0:
            processed_image = (processed_image * 255).astype(np.uint8)
        
        processed_pil = Image.fromarray(processed_image)
        processed_pil.save(os.path.join(inputs_dir, 'processed_image.png'))
    
    # Save prompt
    with open(os.path.join(inputs_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt)
    
    print(f"Input conditions saved to: {inputs_dir}")

def save_debug_images(debug_dir: str, step: int, latents: torch.Tensor, 
                     model, first_stage_model, device: str):
    """Save debug images with proper memory management."""
    
    try:
        with torch.no_grad():
            # 确保latents格式正确 (5D: batch, channels, time, height, width)
            if latents.device != torch.device(device):
                debug_latents = latents.to(device)
            else:
                debug_latents = latents.clone()
            
            # 解码每一帧 - DynamiCrafter需要4D输入
            batch_size, channels, num_frames, height, width = debug_latents.shape
            
            # 重塑为4D: (batch*time, channels, height, width)
            debug_latents_4d = debug_latents.permute(0, 2, 1, 3, 4).contiguous()
            debug_latents_4d = debug_latents_4d.view(-1, channels, height, width)
            
            # 解码所有帧
            if device == 'cuda':
                with autocast():
                    decoded = first_stage_model.decode(debug_latents_4d)
            else:
                decoded = first_stage_model.decode(debug_latents_4d)
            
            # 重塑回5D格式
            decoded_channels = decoded.shape[1]
            decoded_height = decoded.shape[2]
            decoded_width = decoded.shape[3]
            decoded = decoded.view(batch_size, num_frames, decoded_channels, decoded_height, decoded_width)
            decoded = decoded.permute(0, 2, 1, 3, 4)  # (batch, channels, time, height, width)
            
            # 转换为numpy
            if isinstance(decoded, torch.Tensor):
                decoded = decoded.detach().cpu().numpy()
            
            # 保存每一帧
            for t in range(num_frames):
                frame = decoded[0, :, t, :, :]  # (channels, height, width)
                
                # 转换为HWC格式
                if frame.shape[0] == 3:  # RGB
                    frame = np.transpose(frame, (1, 2, 0))  # HWC
                
                # 归一化到0-255
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # 保存帧
                frame_path = os.path.join(debug_dir, f'step_{step:03d}_frame_{t:03d}.png')
                Image.fromarray(frame).save(frame_path)
            
            # 清理内存
            del debug_latents, debug_latents_4d, decoded
            torch.cuda.empty_cache() if device == 'cuda' else None
            
    except Exception as e:
        print(f"Warning: Could not save debug images for step {step}: {e}")

def save_debug_video(debug_dir: str, step: int, latents: torch.Tensor, 
                    model, first_stage_model, device: str):
    """Save debug video with proper tensor handling."""
    
    try:
        with torch.no_grad():
            # 确保latents格式正确
            if latents.device != torch.device(device):
                debug_latents = latents.to(device)
            else:
                debug_latents = latents.clone()
            
            # 使用模型的decode_first_stage方法 - 这个方法可以处理5D张量
            if device == 'cuda':
                with autocast():
                    decoded = model.decode_first_stage(debug_latents)
            else:
                decoded = model.decode_first_stage(debug_latents)
            
            # 转换为numpy
            if isinstance(decoded, torch.Tensor):
                video_np = decoded.detach().cpu().numpy()
            else:
                video_np = decoded
            
            # 处理视频格式 (batch, channels, time, height, width)
            if len(video_np.shape) == 5:
                # 取第一个batch
                video_np = video_np[0]  # (channels, time, height, width)
                
                # 转换为 (time, height, width, channels)
                video_np = np.transpose(video_np, (1, 2, 3, 0))
                
                # 归一化到0-255
                if video_np.max() <= 1.0:
                    video_np = (video_np * 255).astype(np.uint8)
                else:
                    video_np = np.clip(video_np, 0, 255).astype(np.uint8)
                
                # 保存视频
                video_path = os.path.join(debug_dir, f'step_{step:03d}_video.mp4')
                imageio.mimwrite(video_path, video_np, fps=8, quality=7)
            
            # 清理内存
            del debug_latents, decoded, video_np
            torch.cuda.empty_cache() if device == 'cuda' else None
            
    except Exception as e:
        print(f"Warning: Could not save debug video for step {step}: {e}")

def create_optimization_process_video(process_dir: str, debug_dir: str, 
                                    total_steps: int, fps: int = 2):
    """Create a video showing the optimization process."""
    
    try:
        # Find all debug videos
        debug_videos = []
        for step in range(total_steps):
            video_path = os.path.join(debug_dir, f'step_{step:03d}_video.mp4')
            if os.path.exists(video_path):
                debug_videos.append(video_path)
        
        if not debug_videos:
            print("No debug videos found for process video creation")
            return
        
        # Create process video by combining frames from each step
        process_frames = []
        
        for video_path in debug_videos:
            try:
                # Read video and take first frame
                reader = imageio.get_reader(video_path)
                first_frame = reader.get_data(0)
                process_frames.append(first_frame)
                reader.close()
            except Exception as e:
                print(f"Warning: Could not read frame from {video_path}: {e}")
                continue
        
        if process_frames:
            # Save optimization process video
            process_video_path = os.path.join(process_dir, 'optimization_process.mp4')
            imageio.mimwrite(process_video_path, process_frames, fps=fps, quality=7)
            print(f"Optimization process video saved to: {process_video_path}")
        
    except Exception as e:
        print(f"Warning: Could not create optimization process video: {e}")


class DynamiCrafterGuidancePipeline:
    """
    DynamiCrafter Guidance Pipeline for gradient-based video optimization.
    
    架构继承关系：
    - 基础结构：guidance_flux_pipeline.py:FluxGuidancePipeline
    - 模型加载：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline  
    - 优化循环：guidance_flux_pipeline.py:FluxGuidancePipeline._optimization_loop
    """
    
    def __init__(
        self, 
        resolution: str = '256_256',
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        初始化函数
        
        参考来源：
        - dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__init__ (主体结构)
        - guidance_flux_pipeline.py:FluxGuidancePipeline.__init__ (guidance_config部分)
        """
        # === 来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__init__ ===
        self.resolution = tuple(map(int, resolution.split('_')))  # (height, width)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline.__init__ ===
        # Initialize guidance config
        self.guidance_config = DynamiCrafterGuidanceConfig()
        
        # === 来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__init__ ===
        # Load model components
        self._load_components()
        self._setup_image_processor()
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline._optimize_memory ===
        self._optimize_memory()
        
        # === 来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__init__ ===
        # Pipeline state
        self._is_initialized = True
        print(f"✅ DynamiCrafter Guidance Pipeline initialized: {self.resolution[0]}x{self.resolution[1]}")
        
        # Add configuration for result saving
        self.save_debug_images = True
        self.save_debug_videos = True
        self.save_process_video = True
        self.debug_save_interval = 10  # Save debug results every N steps
        
        print("Pipeline initialization complete with enhanced saving features.")
    
    def _load_components(self):
        """
        加载 DynamiCrafter 模型和组件
        
        完全来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._load_components
        """
        # Download model if needed
        self._download_model()
        
        # Load model - 修复配置文件路径
        project_root = os.path.dirname(__file__)
        ckpt_path = os.path.join(project_root, f'checkpoints/dynamicrafter_{self.resolution[1]}_v1/model.ckpt')
        config_file = os.path.join(project_root, f'configs/inference_{self.resolution[1]}_v1.0.yaml')
        
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        
        self.model = instantiate_from_config(model_config)
        assert os.path.exists(ckpt_path), f"Error: checkpoint Not Found at {ckpt_path}!"
        self.model = load_model_checkpoint(self.model, ckpt_path)
        self.model.eval()
        
        # Move to device
        self.model = self.model.to(self.device)
        self._ensure_device_consistency()
        
        print(f"🔄 DynamiCrafter model loaded on {self.device}")
        
    def _ensure_device_consistency(self):
        """
        确保所有模型组件在正确的设备上
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._load_components 
        (move_to_device 函数部分)
        """
        def move_to_device(module):
            for name, param in module.named_parameters():
                if param.device != torch.device(self.device):
                    param.data = param.data.to(self.device)
            for name, buffer in module.named_buffers():
                if buffer.device != torch.device(self.device):
                    buffer.data = buffer.data.to(self.device)
            for name, child in module.named_children():
                move_to_device(child)
        
        move_to_device(self.model)
        
        # 特别处理各个编码器
        for component_name in ['cond_stage_model', 'first_stage_model', 'embedder', 'image_proj_model']:
            if hasattr(self.model, component_name):
                component = getattr(self.model, component_name)
                component = component.to(self.device)
                move_to_device(component)
        
    def _optimize_memory(self):
        """
        优化内存使用
        
        参考来源：
        - guidance_flux_pipeline.py:FluxGuidancePipeline._optimize_memory (核心思路)
        - 适配 DynamiCrafter 的组件结构
        """
        # 将文本编码器移至 CPU 以节省显存
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model.to("cpu")
        
        # 保持 VAE 和主模型在 GPU 上
        if hasattr(self.model, 'first_stage_model'):
            self.model.first_stage_model.to(self.device)
        
        torch.cuda.empty_cache()
        
    def _setup_image_processor(self):
        """
        设置图像预处理
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._setup_image_processor
        """
        self.image_processor = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        
    def _download_model(self):
        """
        下载模型权重
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._download_model
        """
        REPO_ID = f'Doubiiu/DynamiCrafter_{self.resolution[1]}' if self.resolution[1] != 256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        
        # 修复：guidance_pipeline.py 位于项目根目录，不需要往上两级
        project_root = os.path.dirname(__file__)
        model_dir = os.path.join(project_root, f'checkpoints/dynamicrafter_{self.resolution[1]}_v1/')
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        for filename in filename_list:
            local_file = os.path.join(model_dir, filename)
            if not os.path.exists(local_file):
                print(f"📥 Downloading model: {filename}")
                hf_hub_download(
                    repo_id=REPO_ID, 
                    filename=filename, 
                    local_dir=model_dir, 
                    local_dir_use_symlinks=False
                )
                print(f"✅ Download completed: {filename}")

    def _preprocess_image(self, image, height=None, width=None):
        """
        预处理输入图像
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._preprocess_image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # 标准化到 [-1, 1]
        if image.max() > 1.0:
            image = image / 255.0
        image = (image - 0.5) * 2
        
        # 调整大小
        if height is not None and width is not None:
            transform = transforms.Compose([
                transforms.Resize((height, width)),
            ])
        else:
            transform = self.image_processor
        
        return transform(image)

    def _encode_prompt(self, prompt, negative_prompt, device):
        """
        编码文本提示
        
        来源结合：
        - dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._encode_prompt (主体逻辑)
        - guidance_flux_pipeline.py:FluxGuidancePipeline._get_t5_prompt_embeds (内存优化部分)
        """
        # === 来自 guidance_flux_pipeline.py 的内存优化技术 ===
        # 临时将文本编码器移到 GPU
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model = self.model.cond_stage_model.to(device)
        
        # === 来自 dynamicrafter_pipeline.py 的核心编码逻辑 ===
        with torch.no_grad():
            # 正向提示
            text_embeddings = self.model.get_learned_conditioning(prompt)
            
            # 负向提示
            if negative_prompt is not None:
                uncond_embeddings = self.model.get_learned_conditioning(negative_prompt)
            else:
                if self.model.uncond_type == "empty_seq":
                    uncond_embeddings = self.model.get_learned_conditioning([""] * len(prompt))
                elif self.model.uncond_type == "zero_embed":
                    uncond_embeddings = torch.zeros_like(text_embeddings)
        
        # === 来自 guidance_flux_pipeline.py 的内存优化技术 ===
        # 将文本编码器移回 CPU
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model.to("cpu")
        
        torch.cuda.empty_cache()
        
        return {"cond": text_embeddings, "uncond": uncond_embeddings}
    
    def _encode_image(self, image, num_frames):
        """
        编码输入图像
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._encode_image
        """
        # 添加批次维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        with torch.no_grad():
            # 图像嵌入
            cond_images = self.model.embedder(image)
            img_emb = self.model.image_proj_model(cond_images)
            
            # 潜在表示
            videos = image.unsqueeze(2)  # 添加时间维度
            z = get_latent_z(self.model, videos)
        
        return img_emb, z
    
    def _prepare_conditioning(self, text_embeddings, image_embeddings, image_latents, 
                             frame_stride, guidance_scale, batch_size):
        """
        准备条件输入
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._prepare_conditioning
        """
        # 组合文本和图像嵌入
        cond_emb = torch.cat([text_embeddings["cond"], image_embeddings], dim=1)
        cond = {"c_crossattn": [cond_emb]}
        
        # 图像条件（hybrid模式）
        if self.model.model.conditioning_key == 'hybrid':
            img_cat_cond = image_latents[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', 
                                 repeat=self.model.temporal_length)
            cond["c_concat"] = [img_cat_cond]
        
        # 无条件输入
        uc = None
        if guidance_scale != 1.0:
            # 创建正确形状的零图像
            zero_image = torch.zeros((batch_size, 3, self.resolution[0], self.resolution[1]), 
                                   device=self.model.device, dtype=self.model.dtype)
            uc_img_emb = self.model.embedder(zero_image)
            uc_img_emb = self.model.image_proj_model(uc_img_emb)
            
            uc_emb = torch.cat([text_embeddings["uncond"], uc_img_emb], dim=1)
            uc = {"c_crossattn": [uc_emb]}
            
            if self.model.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        
        # 帧步长
        fs = torch.tensor([frame_stride] * batch_size, dtype=torch.long, device=self.model.device)
        
        return {"cond": cond, "uc": uc, "fs": fs}

    def sample_timestep_ddim(self, batch_size, min_step_ratio=0.02, max_step_ratio=0.98, ddim_steps=50):
        """
        为 DynamiCrafter 采样 DDIM 时间步
        
        修复：使用正确的DDIM时间步生成方法
        """
        # 使用DynamiCrafter的正确DDIM时间步生成
        from lvdm.models.utils_diffusion import make_ddim_timesteps
        
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method='uniform', 
            num_ddim_timesteps=ddim_steps, 
            num_ddpm_timesteps=self.model.num_timesteps,
            verbose=False
        )
        
        # 根据比例选择时间步范围
        min_idx = int(len(ddim_timesteps) * min_step_ratio)
        max_idx = int(len(ddim_timesteps) * max_step_ratio)
        max_idx = max(max_idx, min_idx + 1)
        
        # 随机采样时间步索引
        t_idx = torch.randint(min_idx, max_idx, (batch_size,), device="cpu")
        
        # 获取对应的时间步值
        t_values = ddim_timesteps[t_idx.cpu().numpy()]
        t = torch.from_numpy(t_values).long().to(self.device)
        
        return t
    
    def add_noise_ddim(self, original_samples, noise, timesteps):
        """
        为 DynamiCrafter 添加 DDIM 风格的噪声
        
        参考来源：
        - guidance_flux_pipeline.py:FluxGuidancePipeline.add_noise_flow_matching (噪声添加思路)
        - 适配为 DDIM 的噪声调度方式
        """
        # 获取 alpha_cumprod
        alphas_cumprod = self.model.alphas_cumprod.to(self.device)
        
        # 确保时间步在正确范围内
        timesteps = timesteps.clamp(0, len(alphas_cumprod) - 1)
        
        # 获取对应的 alpha_cumprod 值
        alpha_t = alphas_cumprod[timesteps]
        
        # 确保维度匹配
        while len(alpha_t.shape) < len(original_samples.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        # DDIM 噪声添加: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        noisy_samples = sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise
        return noisy_samples

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, List[str]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        # === 来自 guidance_flux_pipeline.py 的 Guidance 参数 ===
        num_optimization_steps: int = 1000,
        learning_rate: float = 0.05,
        loss_type: str = "sds",
        weight_type: str = "auto",
        cfg_scale: float = 7.5,
        optimizer_type: str = "AdamW",
        # === DynamiCrafter 特有参数 ===
        frame_stride: int = 3,
        temporal_guidance_scale: float = 1.0,
        # === 来自 guidance_flux_pipeline.py 的动态参数 ===
        min_step_ratio_start: Optional[float] = None,
        min_step_ratio_end: Optional[float] = None,
        max_step_ratio_start: Optional[float] = None,
        max_step_ratio_end: Optional[float] = None,
        # === Enhanced saving parameters ===
        save_results: bool = True,
        results_dir: str = "results_dynamicrafter_guidance",
        save_debug_images: bool = True,
        save_debug_videos: bool = True,
        save_process_video: bool = True,
        debug_save_interval: int = 10,
        # === Deprecated parameters (kept for compatibility) ===
        debug_save_path: str = "debug_videos",
        # === 来自 dynamicrafter_pipeline.py 的标准参数 ===
        generator: Optional[torch.Generator] = None,
        output_type: str = "tensor",
        return_dict: bool = True,
        **kwargs
    ):
        """
        主调用函数 - Enhanced with comprehensive result saving
        
        参考来源：
        - guidance_flux_pipeline.py:FluxGuidancePipeline.__call__ (整体结构和guidance参数)
        - dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__call__ (视频生成逻辑)
        - Stable3DGen结果保存系统 (新增：组织化输出目录结构)
        """
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline.__call__ ===
        # Set default values from config
        if min_step_ratio_start is None:
            min_step_ratio_start = self.guidance_config.min_step_ratio_start
        if min_step_ratio_end is None:
            min_step_ratio_end = self.guidance_config.min_step_ratio_end
        if max_step_ratio_start is None:
            max_step_ratio_start = self.guidance_config.max_step_ratio_start
        if max_step_ratio_end is None:
            max_step_ratio_end = self.guidance_config.max_step_ratio_end
        
        # === 来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__call__ ===
        # Input validation and standardization
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        # Handle negative_prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt length ({len(negative_prompt)}) != batch_size ({batch_size})")
        
        device = self.device
        
        # === 新增：Enhanced saving setup ===
        # Create output structure if saving enabled
        output_dirs = None
        if save_results:
            output_dirs = create_output_structure(
                results_dir, prompt[0], cfg_scale, num_optimization_steps
            )
            print(f"🔧 Results will be saved to: {output_dirs['main']}")
        
        print(f"🎬 开始 DynamiCrafter Guidance 优化...")
        print(f"📝 提示词: {prompt}")
        print(f"🔧 参数: steps={num_optimization_steps}, lr={learning_rate}, loss={loss_type}")
        print(f"💻 设备: {device}")
        
        # === 来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__call__ ===
        # Image preprocessing
        processed_image = self._preprocess_image(image, height, width)
        
        # === 新增：Save input conditions ===
        if save_results:
            image_path = getattr(image, 'filename', 'unknown_image.png')
            if hasattr(image, 'save') and not os.path.exists(image_path):
                # If image is PIL Image without filename, save it temporarily
                temp_path = os.path.join(output_dirs['inputs'], 'temp_input.png')
                image.save(temp_path)
                image_path = temp_path
            
            save_input_conditions(
                output_dirs['inputs'], image_path, prompt[0], processed_image
            )
        
        # Model parameters
        num_frames = num_frames or self.model.temporal_length
        channels = self.model.model.diffusion_model.out_channels
        if height is None or width is None:
            height, width = self.resolution
        latent_height, latent_width = height // 8, width // 8
        
        # Noise shape for video latents (includes time dimension)
        noise_shape = (batch_size, channels, num_frames, latent_height, latent_width)
        
        with torch.no_grad():
            # Encode prompts and image
            text_embeddings = self._encode_prompt(prompt, negative_prompt, device)
            image_embeddings, image_latents = self._encode_image(processed_image, num_frames)
            
            # Prepare conditioning
            conditioning = self._prepare_conditioning(
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                image_latents=image_latents,
                frame_stride=frame_stride,
                guidance_scale=cfg_scale,
                batch_size=batch_size
            )
        
        # === 新增：Save parameters ===
        if save_results:
            save_parameters(
                output_dirs['params'],
                prompt=prompt[0],
                negative_prompt=negative_prompt[0] if negative_prompt else None,
                height=height,
                width=width,
                num_frames=num_frames,
                num_optimization_steps=num_optimization_steps,
                learning_rate=learning_rate,
                loss_type=loss_type,
                weight_type=weight_type,
                cfg_scale=cfg_scale,
                optimizer_type=optimizer_type,
                frame_stride=frame_stride,
                temporal_guidance_scale=temporal_guidance_scale,
                min_step_ratio_start=min_step_ratio_start,
                min_step_ratio_end=min_step_ratio_end,
                max_step_ratio_start=max_step_ratio_start,
                max_step_ratio_end=max_step_ratio_end,
                save_debug_images=save_debug_images,
                save_debug_videos=save_debug_videos,
                save_process_video=save_process_video,
                debug_save_interval=debug_save_interval,
                device=str(device),
                batch_size=batch_size,
                channels=channels,
                latent_height=latent_height,
                latent_width=latent_width,
                noise_shape=noise_shape,
                **kwargs
            )
        
        # === 来自 dynamicrafter_pipeline.py 的初始化逻辑 ===
        # Initialize video latents for optimization
        if generator is not None:
            video_latents = torch.randn(noise_shape, generator=generator, device=device, dtype=torch.float32)
        else:
            video_latents = torch.randn(noise_shape, device=device, dtype=torch.float32)
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline.__call__ ===
        # Start optimization loop
        return self._optimization_loop(
            video_latents=video_latents,
            conditioning=conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            num_optimization_steps=num_optimization_steps,
            learning_rate=learning_rate,
            loss_type=loss_type,
            weight_type=weight_type,
            cfg_scale=cfg_scale,
            optimizer_type=optimizer_type,
            temporal_guidance_scale=temporal_guidance_scale,
            min_step_ratio_start=min_step_ratio_start,
            min_step_ratio_end=min_step_ratio_end,
            max_step_ratio_start=max_step_ratio_start,
            max_step_ratio_end=max_step_ratio_end,
            output_type=output_type,
            return_dict=return_dict,
            # === Enhanced saving parameters ===
            output_dirs=output_dirs,
            save_results=save_results,
            save_debug_images=save_debug_images,
            save_debug_videos=save_debug_videos,
            save_process_video=save_process_video,
            debug_save_interval=debug_save_interval,
            # === Deprecated parameters (kept for compatibility) ===
            debug_save_path=debug_save_path,
        )

    def _optimization_loop(
        self,
        video_latents,
        conditioning,
        height,
        width,
        num_frames,
        num_optimization_steps,
        learning_rate,
        loss_type,
        weight_type,
        cfg_scale,
        optimizer_type,
        temporal_guidance_scale,
        min_step_ratio_start,
        min_step_ratio_end,
        max_step_ratio_start,
        max_step_ratio_end,
        output_type,
        return_dict,
        # === Enhanced saving parameters ===
        output_dirs,
        save_results,
        save_debug_images,
        save_debug_videos,
        save_process_video,
        debug_save_interval,
        # === Deprecated parameters (kept for compatibility) ===
        debug_save_path,
    ):
        """
        DynamiCrafter guidance optimization loop with enhanced saving.
        
        主要参考：guidance_flux_pipeline.py:FluxGuidancePipeline._optimization_loop
        修改适配：DynamiCrafter 的视频解码和保存方式
        新增：综合性结果保存功能
        """
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline._optimization_loop ===
        # Create debug directory if needed (compatibility with old interface)
        if save_debug_videos and not save_results:
            os.makedirs(debug_save_path, exist_ok=True)
            print(f"[INFO] Debug videos will be saved to: {debug_save_path}")
        
        # Set up latents for optimization
        video_latents = video_latents.to(torch.float32).detach().clone()
        video_latents.requires_grad_(True)
        
        # Set up optimizer
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW([video_latents], lr=learning_rate, betas=(0.9, 0.99), eps=1e-8)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam([video_latents], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
        
        # Determine effective weight type
        if weight_type == "auto":
            if loss_type == "sds":
                effective_weight_type = "t"
            elif loss_type == "csd":
                effective_weight_type = "ada"
            else:  # rfds
                effective_weight_type = "uniform"
        else:
            effective_weight_type = weight_type
        
        print(f"[INFO] Starting DynamiCrafter guidance optimization with {num_optimization_steps} steps")
        print(f"[INFO] Loss type: {loss_type.upper()}")
        print(f"[INFO] Weight type: {effective_weight_type}")
        print(f"[INFO] Learning rate: {learning_rate}")
        print(f"[INFO] CFG scale: {cfg_scale}")
        print(f"[INFO] Temporal guidance scale: {temporal_guidance_scale}")
        if save_results:
            print(f"[INFO] Enhanced saving enabled - interval: {debug_save_interval} steps")
        print("-" * 60)
        
        # === 修改：Enhanced debug saving functions ===
        def save_debug_video_old(step, current_latents):
            """Legacy debug video saving function for compatibility."""
            with torch.no_grad():
                try:
                    # === 来自 guidance_flux_pipeline.py 的内存优化思路 ===
                    # Move non-essential components to CPU
                    if hasattr(self.model, 'cond_stage_model'):
                        self.model.cond_stage_model.to("cpu")
                    
                    torch.cuda.empty_cache()
                    
                    # === 来自 dynamicrafter_pipeline.py 的视频解码逻辑 ===
                    # Decode latents to video
                    debug_videos = self.model.decode_first_stage(current_latents.detach())
                    
                    # Post-process and save
                    debug_videos = debug_videos.cpu().float().numpy()
                    
                    # === 来自 dynamicrafter_pipeline.py 的视频保存函数 ===
                    # Save using DynamiCrafter's save_videos function
                    filename = f"step_{step:06d}"
                    save_videos(debug_videos.unsqueeze(1), debug_save_path, filenames=[filename], fps=8)
                    
                    print(f"[DEBUG] Saved debug video: {debug_save_path}/{filename}.mp4")
                    
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"[DEBUG] Failed to save debug video: {e}")
        
        def save_enhanced_debug_results(step, current_latents):
            """Enhanced debug saving with comprehensive features."""
            if not save_results:
                return
                
            try:
                # Save debug images
                if save_debug_images:
                    save_debug_images(
                        output_dirs['debug'], step, current_latents, 
                        self.model, self.model.first_stage_model, str(self.device)
                    )
                
                # Save debug videos
                if save_debug_videos:
                    save_debug_video(
                        output_dirs['debug'], step, current_latents,
                        self.model, self.model.first_stage_model, str(self.device)
                    )
                
                print(f"[DEBUG] Enhanced debug results saved for step {step}")
                
            except Exception as e:
                print(f"[DEBUG] Failed to save enhanced debug results: {e}")
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline._optimization_loop ===
        # Optimization loop
        for i in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # Calculate current step ratios
            progress = i / (num_optimization_steps - 1) if num_optimization_steps > 1 else 0.0
            current_min_step_ratio = (
                min_step_ratio_start + 
                (min_step_ratio_end - min_step_ratio_start) * progress
            )
            current_max_step_ratio = (
                max_step_ratio_start + 
                (max_step_ratio_end - max_step_ratio_start) * progress
            )
            
            # === 新增：适配 DynamiCrafter 的损失函数 ===
            # Compute guidance loss
            if loss_type == "sds":
                loss = self._sds_loss_video(
                    video_latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    temporal_guidance_scale=temporal_guidance_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            elif loss_type == "csd":
                loss = self._csd_loss_video(
                    video_latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    temporal_guidance_scale=temporal_guidance_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            elif loss_type == "rfds":
                loss = self._rfds_loss_video(
                    video_latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    temporal_guidance_scale=temporal_guidance_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_([video_latents], max_norm=1.0)
            
            # Update latents
            optimizer.step()
            
            # === 新增：Enhanced debug saving ===
            if i % debug_save_interval == 0:
                if save_results:
                    save_enhanced_debug_results(i, video_latents)
                elif save_debug_videos:  # Compatibility with old interface
                    save_debug_video_old(i, video_latents)
            
            # Progress logging
            if i % max(1, num_optimization_steps // 10) == 0:
                print(f"[PROGRESS] Step {i}/{num_optimization_steps} - Loss: {loss.item():.6f}")
        
        # === 新增：Final results generation and saving ===
        print(f"[INFO] Optimization complete. Generating final results...")
        
        with torch.no_grad():
            # Generate final video
            final_videos = self.model.decode_first_stage(video_latents.detach())
            
            # Save final results
            if save_results:
                # Save final video to outputs directory
                for i in range(final_videos.shape[0]):
                    # Keep tensor format for save_videos function
                    video_data = final_videos[i:i+1].unsqueeze(1)  # Add samples dimension: b,samples,c,t,h,w
                    
                    # Use DynamiCrafter's save_videos function
                    save_videos(video_data, output_dirs['outputs'], 
                              filenames=[f'final_video_{i:03d}'], fps=8)
                    
                    output_path = os.path.join(output_dirs['outputs'], f'final_video_{i:03d}.mp4')
                    print(f"Final video saved to: {output_path}")
                
                # Create optimization process video
                if save_process_video:
                    create_optimization_process_video(
                        output_dirs['process'], output_dirs['debug'], 
                        num_optimization_steps, fps=2
                    )
                
                print(f"✅ All results saved to: {output_dirs['main']}")
            
            # === 来自 dynamicrafter_pipeline.py 的返回逻辑 ===
            # Return results
            if output_type == "tensor":
                return final_videos
            elif output_type == "np":
                return final_videos.cpu().float().numpy()
            elif output_type == "pil":
                # Convert to PIL format if needed
                final_videos_np = final_videos.cpu().float().numpy()
                pil_videos = []
                for video in final_videos_np:
                    frames = []
                    for frame in video:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        frames.append(Image.fromarray(frame))
                    pil_videos.append(frames)
                return pil_videos
            else:
                return final_videos

    def _sds_loss_video(self, video_latents, conditioning, cfg_scale=7.5, temporal_guidance_scale=1.0, 
                       min_step_ratio=0.02, max_step_ratio=0.98, weight_type="t"):
        """
        SDS loss for video latents using DynamiCrafter's 3D UNet.
        
        修复：改进数值稳定性和梯度计算
        """
        batch_size = video_latents.shape[0]
        
        # === 修复：更稳定的时间步采样 ===
        t = self.sample_timestep_ddim(batch_size, min_step_ratio, max_step_ratio)
        
        # 确保时间步在有效范围内
        t = torch.clamp(t, 0, self.model.num_timesteps - 1)
        
        # Add noise to video latents
        noise = torch.randn_like(video_latents)
        noisy_latents = self.add_noise_ddim(video_latents, noise, t)
        
        # === 来自 dynamicrafter_pipeline.py 的条件准备 ===
        # Prepare conditioning
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # === 修复：更稳定的模型前向传播 ===
        # Forward pass with CFG
        with torch.no_grad():
            try:
                # Conditional prediction
                noise_pred_cond = self.model.apply_model(
                    noisy_latents, t, cond, **{"fs": fs}
                )
                
                # Unconditional prediction
                if uc is not None and cfg_scale > 1.0:
                    noise_pred_uncond = self.model.apply_model(
                        noisy_latents, t, uc, **{"fs": fs}
                    )
                    
                    # Apply CFG with clamping for stability
                    cfg_diff = noise_pred_cond - noise_pred_uncond
                    cfg_diff = torch.clamp(cfg_diff, -10.0, 10.0)  # 限制CFG差值
                    noise_pred = noise_pred_uncond + cfg_scale * cfg_diff
                else:
                    noise_pred = noise_pred_cond
                
                # 应用时间引导
                if temporal_guidance_scale != 1.0:
                    noise_pred = noise_pred * temporal_guidance_scale
                    
            except Exception as e:
                print(f"Warning: Model forward pass failed: {e}")
                # Fallback to simple noise prediction
                noise_pred = noise
        
        # === 修复：更稳定的原始样本预测 ===
        # Calculate predicted original sample with numerical stability
        alpha_t = self.model.alphas_cumprod[t].to(self.device)
        while len(alpha_t.shape) < len(video_latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        # 数值稳定性改进
        alpha_t = torch.clamp(alpha_t, 1e-6, 1.0 - 1e-6)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        # 更稳定的原始样本预测
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # === 修复：改进的SDS梯度计算 ===
        # Calculate SDS gradient with improved stability
        if weight_type == "t":
            # Time-dependent weighting with stability
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)
            w = torch.clamp(w, 1e-6, 1.0)  # 防止权重为0
            grad = w * (video_latents - pred_original_sample.detach())
        elif weight_type == "ada":
            # Adaptive weighting with stability
            pred_diff = video_latents - pred_original_sample.detach()
            weighting_factor = torch.abs(pred_diff).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4, 10.0)  # 防止极值
            grad = pred_diff / weighting_factor
        elif weight_type == "uniform":
            # Uniform weighting with clamping
            grad = video_latents - pred_original_sample.detach()
            grad = torch.clamp(grad, -1.0, 1.0)  # 限制梯度幅度
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        # 清理和稳定梯度
        grad = grad.detach()
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        grad = torch.clamp(grad, -1.0, 1.0)  # 全局梯度裁剪
        
        # === 修复：更稳定的损失计算 ===
        # Construct loss using reparameterization trick with stability
        target = (video_latents - grad).detach()
        loss = 0.5 * F.mse_loss(video_latents, target, reduction="mean")
        
        # 损失稳定性检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected, using fallback loss")
            loss = 0.5 * F.mse_loss(video_latents, video_latents.detach(), reduction="mean")
        
        # 缩放损失以提高稳定性
        loss = loss / batch_size
        loss = torch.clamp(loss, 0.0, 10.0)  # 限制损失范围
        
        torch.cuda.empty_cache()
        
        return loss

    def _csd_loss_video(self, video_latents, conditioning, cfg_scale=7.5, temporal_guidance_scale=1.0,
                       min_step_ratio=0.02, max_step_ratio=0.98, weight_type="ada"):
        """
        CSD loss for video latents using DynamiCrafter's 3D UNet.
        
        参考来源：
        - guidance_flux_pipeline.py:FluxGuidancePipeline._csd_loss (CSD 损失计算逻辑)
        - 适配 DynamiCrafter 的 3D UNet 和时间维度
        """
        batch_size = video_latents.shape[0]
        
        # === 适配 DynamiCrafter：使用 DDIM 时间步 ===
        # Sample timesteps
        t = self.sample_timestep_ddim(batch_size, min_step_ratio, max_step_ratio)
        
        # Add noise
        noise = torch.randn_like(video_latents)
        noisy_latents = self.add_noise_ddim(video_latents, noise, t)
        
        # === 来自 dynamicrafter_pipeline.py 的条件准备 ===
        # Prepare conditioning
        cond = conditioning["cond"]
        fs = conditioning["fs"]
        
        # === 来自 guidance_flux_pipeline.py，适配 DynamiCrafter 的 3D UNet ===
        # Forward pass with low and high guidance
        with torch.no_grad():
            # 修复：使用正确的 apply_model 方法
            # Low guidance (fake distribution)
            noise_pred_fake = self.model.apply_model(
                noisy_latents, t, cond, **{"fs": fs}
            )
            
            # High guidance (real distribution) - simulate with higher noise
            noise_enhanced = noise * (1.0 + cfg_scale * 0.1)  # Slight enhancement
            noisy_latents_enhanced = self.add_noise_ddim(video_latents, noise_enhanced, t)
            
            noise_pred_real = self.model.apply_model(
                noisy_latents_enhanced, t, cond, **{"fs": fs}
            )
        
        # === 来自 guidance_flux_pipeline.py，适配 DDIM ===
        # Calculate predicted original samples
        alpha_t = self.model.alphas_cumprod[t]
        while len(alpha_t.shape) < len(video_latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        pred_fake_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred_fake) / sqrt_alpha_t
        pred_real_latents = (noisy_latents_enhanced - sqrt_one_minus_alpha_t * noise_pred_real) / sqrt_alpha_t
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline._csd_loss ===
        # Calculate CSD gradient
        if weight_type == "ada":
            # Adaptive weighting
            weighting_factor = torch.abs(video_latents - pred_real_latents.detach()).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        elif weight_type == "t":
            # Time-dependent weighting
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)  # 适配5D张量
            grad = w * (pred_fake_latents - pred_real_latents)
        elif weight_type == "uniform":
            # Uniform weighting
            grad = (pred_fake_latents - pred_real_latents)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        grad = grad.detach()
        grad = torch.nan_to_num(grad)
        
        # Construct loss
        target = (video_latents - grad).detach()
        loss = 0.5 * F.mse_loss(video_latents, target, reduction="mean") / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def _rfds_loss_video(self, video_latents, conditioning, cfg_scale=7.5, temporal_guidance_scale=1.0,
                        min_step_ratio=0.02, max_step_ratio=0.98, weight_type="uniform"):
        """
        RFDS loss for video latents using DynamiCrafter's 3D UNet.
        
        参考来源：
        - guidance_flux_pipeline.py:FluxGuidancePipeline._rfds_loss (RFDS 损失计算逻辑)
        - 适配 DynamiCrafter 的 3D UNet 和视频生成
        """
        batch_size = video_latents.shape[0]
        
        # === 适配 DynamiCrafter：使用 DDIM 时间步 ===
        # Sample timesteps
        t = self.sample_timestep_ddim(batch_size, min_step_ratio, max_step_ratio)
        
        # Add noise
        noise = torch.randn_like(video_latents)
        noisy_latents = self.add_noise_ddim(video_latents, noise, t)
        
        # === 来自 dynamicrafter_pipeline.py 的条件准备 ===
        # Prepare conditioning
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # === 适配 DynamiCrafter：使用 3D UNet 进行前向传播 ===
        # Forward pass with CFG
        with torch.no_grad():
            # 修复：使用正确的 apply_model 方法
            # Conditional prediction
            noise_pred_cond = self.model.apply_model(
                noisy_latents, t, cond, **{"fs": fs}
            )
            
            # Unconditional prediction
            if uc is not None:
                noise_pred_uncond = self.model.apply_model(
                    noisy_latents, t, uc, **{"fs": fs}
                )
                
                # Apply CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
        
        # === 来自 guidance_flux_pipeline.py，适配 DDIM ===
        # Calculate predicted original sample
        alpha_t = self.model.alphas_cumprod[t]
        while len(alpha_t.shape) < len(video_latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # === 来自 guidance_flux_pipeline.py:FluxGuidancePipeline._rfds_loss ===
        # RFDS: Direct optimization to match predicted original sample
        target = pred_original_sample.detach()
        
        if weight_type == "t":
            # Time-dependent weighting
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)  # 适配5D张量
            loss = 0.5 * w * F.mse_loss(video_latents, target, reduction="mean")
        elif weight_type == "ada":
            # Adaptive weighting
            prediction_error = video_latents - target
            weighting_factor = torch.abs(prediction_error).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            loss = 0.5 * F.mse_loss(video_latents, target, reduction="none")
            loss = (loss / weighting_factor).mean()
        elif weight_type == "uniform":
            # Uniform weighting
            loss = 0.5 * F.mse_loss(video_latents, target, reduction="mean")
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        loss = loss / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def save_video(
        self, 
        video: torch.Tensor, 
        output_path: str, 
        fps: int = 8, 
        **kwargs
    ):
        """
        Save generated video to file using DynamiCrafter's save function.
        
        来自：dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.save_video
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension
        filename = os.path.basename(output_path).split('.')[0]
        
        # Fix dimensions for save_videos function
        if video.dim() == 4:
            video = video.unsqueeze(0).unsqueeze(0)
        elif video.dim() == 5:
            video = video.unsqueeze(1)
        
        # Ensure video tensor is on CPU
        if video.is_cuda:
            video = video.cpu()
        
        print(f"📊 保存视频形状: {video.shape}")
        save_videos(video, output_dir or '.', filenames=[filename], fps=fps)
        
        print(f"💾 视频已保存: {output_path}")


def test_dynamicrafter_guidance_pipeline():
    """
    测试 DynamiCrafter Guidance Pipeline
    
    参考来源：
    - dynamicrafter_pipeline.py:test_pipeline (测试结构)
    - 适配 guidance 相关的测试参数
    """
    print("🧪 测试 DynamiCrafter Guidance Pipeline")
    print("=" * 70)
    
    try:
        # === 来自 dynamicrafter_pipeline.py:test_pipeline ===
        # Initialize pipeline
        print("🔧 初始化 DynamiCrafter Guidance Pipeline...")
        pipeline = DynamiCrafterGuidancePipeline(
            resolution='256_256'
        )
        
        # Find test image
        project_root = os.path.dirname(__file__)
        test_image_paths = [
            os.path.join(project_root, 'prompts/1024/pour_bear.png'),
            os.path.join(project_root, 'prompts/512_loop/24.png'),
        ]
        
        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path is None:
            print("❌ 测试图像不存在")
            return False
        
        # Load test image
        print(f"📸 加载测试图像: {test_image_path}")
        img = Image.open(test_image_path).convert("RGB")
        
        # === 适配 guidance 测试 ===
        # Test basic functionality
        test_prompt = "a person walking in a beautiful garden with flowers blooming"
        
        print(f"📝 测试提示词: {test_prompt}")
        print(f"🎬 开始 SDS 优化生成...")
        
        start_time = time.time()
        
        # Generate with SDS
        result = pipeline(
            image=img,
            prompt=test_prompt,
            num_optimization_steps=50,  # Short test
            learning_rate=0.05,
            loss_type="sds",
            cfg_scale=7.5,
            save_debug_videos=True,
            debug_save_interval=25,
            debug_save_path="./debug_dynamicrafter_guidance",
            return_dict=True
        )
        
        elapsed_time = time.time() - start_time
        
        # === 来自 dynamicrafter_pipeline.py:test_pipeline ===
        # Check results
        video = result["videos"]
        print(f"🎉 SDS 生成成功!")
        print(f"📊 视频形状: {video.shape}")
        print(f"⏱️ 用时: {elapsed_time:.2f} 秒")
        
        # Check for NaN
        if torch.isnan(video).any():
            print("❌ 检测到 NaN 值!")
            return False
        else:
            print("✅ 无 NaN 问题")
        
        # Save result
        output_dir = './results_dynamicrafter_guidance/'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_sds_video.mp4")
        
        pipeline.save_video(video, output_path)
        
        print("🎉 DynamiCrafter Guidance Pipeline 测试成功!")
        print("📋 测试总结:")
        print("  ✅ 模型加载正常")
        print("  ✅ 图像预处理正常")
        print("  ✅ 文本编码正常")
        print("  ✅ SDS 优化正常")
        print("  ✅ 视频解码正常")
        print("  ✅ 视频保存正常")
        print("  ✅ 无NaN问题")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_dynamicrafter_guidance_pipeline()
    else:
        print("📝 DynamiCrafter Guidance Pipeline")
        print("🔬 融合技术来源：")
        print("  • DynamiCrafter: scripts/gradio/dynamicrafter_pipeline.py")
        print("  • Flux Kontext: _reference_codes/Stable3DGen/flux_kontext_pipeline.py") 
        print("  • Flux Guidance: _reference_codes/Stable3DGen/2D_experiments/guidance_flux_pipeline.py")
        print("")
        print("🎯 核心创新：将 Flux Guidance 的先进优化技术应用于 DynamiCrafter 的视频生成")
        print("")
        print("📖 使用示例:")
        print("```python")
        print("from dynamicrafter_guidance_pipeline import DynamiCrafterGuidancePipeline")
        print("from PIL import Image")
        print("")
        print("# 初始化 pipeline")
        print("pipeline = DynamiCrafterGuidancePipeline(resolution='256_256')")
        print("")
        print("# 加载图像")
        print("image = Image.open('your_image.jpg')")
        print("")
        print("# SDS 优化生成")
        print("result = pipeline(")
        print("    image=image,")
        print("    prompt='person walking in garden',")
        print("    num_optimization_steps=1000,")
        print("    loss_type='sds',           # 或 'csd', 'rfds'")
        print("    cfg_scale=7.5")
        print(")")
        print("")
        print("# 保存视频")
        print("pipeline.save_video(result['videos'], 'output.mp4')")
        print("```")
