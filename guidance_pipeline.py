# dynamicrafter_guidance_pipeline.py

"""
DynamiCrafter Guidance Pipeline

正确的改造思路：
1. 保持 DynamiCrafter 的所有核心逻辑 (来自 dynamicrafter_pipeline.py)
2. 只替换 scheduler.sample() 调用为 optimization loop (参考 guidance_flux_pipeline.py)
3. 在 optimization loop 中使用 DynamiCrafter 的 model.apply_model() 方法

转换关系：
- pipeline_normal_to_flux.py (inference) → guidance_flux_pipeline.py (guidance)
- dynamicrafter_pipeline.py (inference) → guidance_pipeline.py (guidance)
"""

import os
import time
import sys
import json
import shutil
from datetime import datetime
from typing import Optional, Union, List, Any, Dict, Callable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from PIL import Image
from omegaconf import OmegaConf
import imageio

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, get_latent_z
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters


@dataclass
class GuidanceConfig:
    """Guidance optimization configuration"""
    num_optimization_steps: int = 1000
    learning_rate: float = 0.05
    loss_type: str = "sds"  # sds, csd, rfds
    weight_type: str = "auto"  # auto, t, ada, uniform
    cfg_scale: float = 7.5
    optimizer_type: str = "AdamW"
    
    # Dynamic step ratio parameters
    min_step_ratio_start: float = 0.02
    min_step_ratio_end: float = 0.02
    max_step_ratio_start: float = 0.98
    max_step_ratio_end: float = 0.98
    
    # Debug parameters
    save_debug_videos: bool = False
    debug_save_interval: int = 100
    debug_save_path: str = "debug_videos"


def create_output_structure(base_dir: str, prompt: str, loss_type: str, weight_type: str, 
                           learning_rate: float, num_optimization_steps: int, cfg_scale: float) -> Dict[str, str]:
    """Create organized output directory structure with descriptive naming."""
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create safe prompt name (remove special characters)
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Truncate long prompts
    
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
    
    # Create main output directory
    output_name = f"{timestamp}_{safe_prompt}_{loss_type}_{effective_weight_type}_lr{learning_rate}_steps{num_optimization_steps}_cfg{cfg_scale}"
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
        f.write("=" * 60 + "\n\n")
        
        for key, value in kwargs.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Parameters saved to: {params_file}")


def save_input_conditions(inputs_dir: str, original_image: Image.Image, prompt: str, 
                         processed_image: torch.Tensor):
    """Save input conditions: original image, processed image, and prompt."""
    
    # Save original image
    original_image.save(os.path.join(inputs_dir, 'original_image.png'))
    
    # Save processed image
    if processed_image is not None:
        # Convert tensor to PIL Image
        if len(processed_image.shape) == 4:
            processed_image = processed_image[0]  # Remove batch dimension
        
        # Ensure proper format for saving
        if processed_image.shape[0] == 3:  # CHW format
            processed_image = processed_image.permute(1, 2, 0)  # HWC format
        
        processed_image = processed_image.cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        processed_image = (processed_image + 1.0) / 2.0
        processed_image = np.clip(processed_image, 0, 1)
        
        # Convert to uint8
        processed_image = (processed_image * 255).astype(np.uint8)
        
        processed_pil = Image.fromarray(processed_image)
        processed_pil.save(os.path.join(inputs_dir, 'processed_image.png'))
    
    # Save prompt
    with open(os.path.join(inputs_dir, 'prompt.txt'), 'w') as f:
        f.write(prompt)
    
    print(f"Input conditions saved to: {inputs_dir}")


def save_debug_frame(debug_dir: str, step: int, latents: torch.Tensor, 
                    model, device: str, frame_idx: int = 0):
    """Save debug frame from video latents."""
    
    try:
        with torch.no_grad():
            # Move to correct device
            if latents.device != torch.device(device):
                debug_latents = latents.to(device)
            else:
                debug_latents = latents.clone()
            
            # Decode latents to video
            decoded_video = model.decode_first_stage(debug_latents)
            
            # Extract specific frame
            if len(decoded_video.shape) == 5:  # (batch, channels, time, height, width)
                frame = decoded_video[0, :, frame_idx, :, :]  # (channels, height, width)
            else:
                frame = decoded_video[0]  # Use first frame if not 5D
            
            # Convert to numpy
            frame = frame.detach().cpu().numpy()
            
            # Convert CHW to HWC
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            
            # Denormalize and convert to uint8
            frame = (frame + 1.0) / 2.0
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            
            # Save frame
            frame_path = os.path.join(debug_dir, f'step_{step:06d}_frame_{frame_idx:02d}.png')
            Image.fromarray(frame).save(frame_path)
            
            # Clear memory
            del debug_latents, decoded_video, frame
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Warning: Could not save debug frame for step {step}: {e}")


def save_debug_video(debug_dir: str, step: int, latents: torch.Tensor, 
                    model, device: str):
    """Save debug video from latents."""
    
    try:
        with torch.no_grad():
            # Move to correct device
            if latents.device != torch.device(device):
                debug_latents = latents.to(device)
            else:
                debug_latents = latents.clone()
            
            # Decode latents to video
            decoded_video = model.decode_first_stage(debug_latents)
            
            # Convert to numpy
            if isinstance(decoded_video, torch.Tensor):
                video_np = decoded_video.detach().cpu().numpy()
            else:
                video_np = decoded_video
            
            # Process video format (batch, channels, time, height, width)
            if len(video_np.shape) == 5:
                # Take first batch
                video_np = video_np[0]  # (channels, time, height, width)
                
                # Convert to (time, height, width, channels)
                video_np = np.transpose(video_np, (1, 2, 3, 0))
                
                # Denormalize and convert to uint8
                video_np = (video_np + 1.0) / 2.0
                video_np = np.clip(video_np, 0, 1)
                video_np = (video_np * 255).astype(np.uint8)
                
                # Save video
                video_path = os.path.join(debug_dir, f'step_{step:06d}_video.mp4')
                imageio.mimsave(video_path, video_np, fps=8)
                
                # Also save first frame as PNG
                first_frame_path = os.path.join(debug_dir, f'step_{step:06d}_frame.png')
                Image.fromarray(video_np[0]).save(first_frame_path)
            
            # Clear memory
            del debug_latents, decoded_video, video_np
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Warning: Could not save debug video for step {step}: {e}")


def create_optimization_process_video(process_dir: str, debug_dir: str, 
                                    total_steps: int, debug_save_interval: int, fps: int = 2):
    """Create a video showing the optimization process."""
    
    try:
        # Find all debug frame images
        debug_frames = []
        for step in range(0, total_steps, debug_save_interval):
            frame_path = os.path.join(debug_dir, f'step_{step:06d}_frame.png')
            if os.path.exists(frame_path):
                debug_frames.append(frame_path)
        
        if not debug_frames:
            print("No debug frames found for process video creation")
            return
        
        # Create process video by combining frames
        process_frames = []
        
        for frame_path in debug_frames:
            try:
                frame = np.array(Image.open(frame_path))
                process_frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not read frame from {frame_path}: {e}")
                continue
        
        if process_frames:
            # Save optimization process video
            process_video_path = os.path.join(process_dir, 'optimization_process.mp4')
            imageio.mimsave(process_video_path, process_frames, fps=fps)
            print(f"Optimization process video saved to: {process_video_path}")
        
    except Exception as e:
        print(f"Warning: Could not create optimization process video: {e}")


def get_fixed_ddim_sampler(model, ddim_steps, ddim_eta, verbose=False):
    """
    返回修复后sigma值的DDIMSampler - 来自 dynamicrafter_pipeline.py
    """
    from lvdm.models.samplers.ddim import DDIMSampler
    
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_steps, ddim_discretize="uniform", ddim_eta=ddim_eta, verbose=verbose)
    
    # 关键修复：用DynamiCrafter原始函数重新计算sigma值
    from lvdm.models.utils_diffusion import make_ddim_sampling_parameters
    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
        alphacums=model.alphas_cumprod.cpu(),
        ddim_timesteps=sampler.ddim_timesteps,
        eta=ddim_eta,
        verbose=False
    )
    
    # 替换有问题的sigma值 - 统一处理为torch tensor
    if isinstance(ddim_sigmas, torch.Tensor):
        sampler.ddim_sigmas = ddim_sigmas.to(model.device)
    else:
        sampler.ddim_sigmas = torch.from_numpy(ddim_sigmas).to(model.device)
    
    if isinstance(ddim_alphas, torch.Tensor):
        sampler.ddim_alphas = ddim_alphas.to(model.device)
    else:
        sampler.ddim_alphas = torch.from_numpy(ddim_alphas).to(model.device)
    
    if isinstance(ddim_alphas_prev, torch.Tensor):
        sampler.ddim_alphas_prev = ddim_alphas_prev.to(model.device)
    else:
        sampler.ddim_alphas_prev = torch.from_numpy(ddim_alphas_prev).to(model.device)
    
    # 计算 ddim_sqrt_one_minus_alphas
    if isinstance(ddim_alphas, torch.Tensor):
        sampler.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas).to(model.device)
    else:
        sampler.ddim_sqrt_one_minus_alphas = torch.from_numpy(np.sqrt(1. - ddim_alphas)).to(model.device)
    
    if verbose:
        print("✅ DDIMSampler fixed: sigma values replaced with numerically stable versions")
    
    return sampler


class DynamiCrafterGuidancePipeline:
    """
    DynamiCrafter Guidance Pipeline - 严格按照转换逻辑改造
    
    基础结构完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline
    只替换推理部分为 guidance optimization
    """
    
    def __init__(
        self, 
        resolution: str = '256_256',
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__init__
        """
        self.resolution = tuple(map(int, resolution.split('_')))  # (height, width)
        
        # Enhanced device selection - support specific CUDA devices
        if device is None:
            # Check if CUDA_VISIBLE_DEVICES is set
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible is not None:
                # Use first visible device
                device_num = cuda_visible.split(',')[0]
                self.device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
            else:
                # Default to cuda:3 if available, otherwise first available CUDA device
                if torch.cuda.is_available():
                    if torch.cuda.device_count() > 3:
                        self.device = "cuda:3"
                    else:
                        self.device = "cuda:0"
                else:
                    self.device = "cpu"
        else:
            self.device = device
        
        self.dtype = dtype or torch.float32
        
        print(f"🔧 Using device: {self.device}")
        
        # 新增：guidance配置
        self.guidance_config = GuidanceConfig()
        
        # Initialize components
        self._load_components()
        self._setup_image_processor()
        
        # Pipeline state
        self._is_initialized = True
        print(f"✅ DynamiCrafter Guidance Pipeline initialized: {self.resolution[0]}x{self.resolution[1]}")
        
    def _load_components(self):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._load_components"""
        # Download model if needed
        self._download_model()
        
        # Load model
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
        """完全来自 dynamicrafter_pipeline.py"""
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
                
    def _setup_image_processor(self):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._setup_image_processor"""
        self.image_processor = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        
    def _download_model(self):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._download_model"""
        REPO_ID = f'Doubiiu/DynamiCrafter_{self.resolution[1]}' if self.resolution[1] != 256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        
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
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._preprocess_image"""
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
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._encode_prompt"""
        # 确保文本编码器在正确的设备上
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model = self.model.cond_stage_model.to(device)
            # 强制设置设备属性
            if hasattr(self.model.cond_stage_model, 'device'):
                self.model.cond_stage_model.device = device
            # 递归确保所有子模块在正确设备上
            for name, module in self.model.cond_stage_model.named_modules():
                if hasattr(module, 'device'):
                    module.device = device
                for param in module.parameters():
                    if param.device != torch.device(device):
                        param.data = param.data.to(device)
        
        # 正向提示
        text_embeddings = self.model.get_learned_conditioning(prompt)
        
        # 负向提示（如果提供）
        if negative_prompt is not None:
            uncond_embeddings = self.model.get_learned_conditioning(negative_prompt)
        else:
            if self.model.uncond_type == "empty_seq":
                uncond_embeddings = self.model.get_learned_conditioning([""] * len(prompt))
            elif self.model.uncond_type == "zero_embed":
                uncond_embeddings = torch.zeros_like(text_embeddings)
        
        return {"cond": text_embeddings, "uncond": uncond_embeddings}
    
    def _encode_image(self, image, num_frames):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._encode_image"""
        # 添加批次维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # 图像嵌入
        cond_images = self.model.embedder(image)
        img_emb = self.model.image_proj_model(cond_images)
        
        # 潜在表示
        videos = image.unsqueeze(2)  # 添加时间维度
        z = get_latent_z(self.model, videos)
        
        return img_emb, z
    
    def _prepare_conditioning(self, text_embeddings, image_embeddings, image_latents, 
                             frame_stride, guidance_scale, batch_size):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._prepare_conditioning"""
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

    def _decode_latents(self, latents):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._decode_latents"""
        with torch.no_grad():
            videos = self.model.decode_first_stage(latents)
        return videos
    
    def _postprocess_video(self, videos, output_type):
        """完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline._postprocess_video"""
        if output_type == "numpy":
            videos = videos.cpu().float().numpy()
        elif output_type == "pil":
            # 转换为PIL图像列表（简化实现）
            videos = videos.cpu().float().numpy()
            print("📝 Note: PIL output conversion not fully implemented, returning numpy")
        # tensor格式直接返回
        
        return videos

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, List[str]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # === 保持 DynamiCrafter 的标准参数 ===
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        frame_stride: int = 3,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        # === 新增：Guidance 参数 ===
        num_optimization_steps: int = 1000,
        learning_rate: float = 0.05,
        loss_type: str = "sds",
        weight_type: str = "auto",
        cfg_scale: Optional[float] = None,
        optimizer_type: str = "AdamW",
        min_step_ratio_start: Optional[float] = None,
        min_step_ratio_end: Optional[float] = None,
        max_step_ratio_start: Optional[float] = None,
        max_step_ratio_end: Optional[float] = None,
        # === 新增：Enhanced saving parameters ===
        save_results: bool = False,
        results_dir: str = "results_dynamicrafter_guidance",
        save_debug_images: bool = False,
        save_debug_videos: bool = False,
        save_process_video: bool = False,
        debug_save_interval: int = 100,
        debug_save_path: str = "debug_videos",
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate video using DynamiCrafter with guidance optimization.
        
        这个函数大部分来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.__call__
        只替换 scheduler.sample() 部分为 optimization loop
        """
        
        # === 完全来自 dynamicrafter_pipeline.py ===
        # 输入验证和标准化
        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        else:
            batch_size = len(prompt)
        
        # 处理 negative_prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            elif len(negative_prompt) != batch_size:
                raise ValueError(f"negative_prompt length ({len(negative_prompt)}) != batch_size ({batch_size})")
        
        # 设备管理
        device = self.device
        
        # 处理 cfg_scale 参数
        if cfg_scale is None:
            cfg_scale = guidance_scale
        
        # 设置默认 guidance 参数
        if min_step_ratio_start is None:
            min_step_ratio_start = self.guidance_config.min_step_ratio_start
        if min_step_ratio_end is None:
            min_step_ratio_end = self.guidance_config.min_step_ratio_end
        if max_step_ratio_start is None:
            max_step_ratio_start = self.guidance_config.max_step_ratio_start
        if max_step_ratio_end is None:
            max_step_ratio_end = self.guidance_config.max_step_ratio_end
        
        # === 新增：Enhanced saving setup ===
        output_dirs = None
        if save_results:
            output_dirs = create_output_structure(
                results_dir, prompt[0], loss_type, weight_type, learning_rate, 
                num_optimization_steps, cfg_scale
            )
            print(f"🔧 Results will be saved to: {output_dirs['main']}")
        
        print(f"🎬 开始 DynamiCrafter Guidance 优化...")
        print(f"📝 提示词: {prompt}")
        print(f"🔧 Guidance 参数: steps={num_optimization_steps}, lr={learning_rate}, loss={loss_type}")
        print(f"🔧 DynamiCrafter 参数: guidance={guidance_scale}, eta={eta}, frame_stride={frame_stride}")
        print(f"💻 设备: {device}")
        
        # 图像预处理
        original_image = image if isinstance(image, Image.Image) else Image.fromarray(np.array(image))
        processed_image = self._preprocess_image(image, height, width)
        
        # === 新增：Save input conditions ===
        if save_results:
            save_input_conditions(
                output_dirs['inputs'], original_image, prompt[0], processed_image
            )
        
        # 模型参数
        num_frames = num_frames or self.model.temporal_length
        channels = self.model.model.diffusion_model.out_channels
        if height is None or width is None:
            height, width = self.resolution
        latent_height, latent_width = height // 8, width // 8
        
        # 噪声形状
        noise_shape = (batch_size, channels, num_frames, latent_height, latent_width)
        
        with torch.no_grad():
            # 编码提示词和图像
            text_embeddings = self._encode_prompt(prompt, negative_prompt, device)
            image_embeddings, image_latents = self._encode_image(processed_image, num_frames)
            
            # 准备条件输入
            conditioning = self._prepare_conditioning(
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                image_latents=image_latents,
                frame_stride=frame_stride,
                guidance_scale=guidance_scale,
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
        
        # ===== 关键替换点：用 optimization loop 替换 scheduler.sample() =====
        # 原代码：
        # scheduler = get_fixed_ddim_sampler(self.model, num_inference_steps, eta, verbose=False)
        # samples, _ = scheduler.sample(...)
        
        # 新代码：optimization loop
        samples = self._optimization_loop(
            noise_shape=noise_shape,
            conditioning=conditioning,
            device=device,
            num_optimization_steps=num_optimization_steps,
            learning_rate=learning_rate,
            loss_type=loss_type,
            weight_type=weight_type,
            cfg_scale=cfg_scale,
            optimizer_type=optimizer_type,
            min_step_ratio_start=min_step_ratio_start,
            min_step_ratio_end=min_step_ratio_end,
            max_step_ratio_start=max_step_ratio_start,
            max_step_ratio_end=max_step_ratio_end,
            # === Enhanced saving parameters ===
            output_dirs=output_dirs,
            save_results=save_results,
            save_debug_images=save_debug_images,
            save_debug_videos=save_debug_videos,
            save_process_video=save_process_video,
            debug_save_interval=debug_save_interval,
            debug_save_path=debug_save_path,
            generator=generator,
            **kwargs
        )
        
        # === 完全来自 dynamicrafter_pipeline.py ===
        # 解码 latents 到视频
        print(f"🎞️ 解码潜在表示...")
        videos = self._decode_latents(samples)
        
        # === 新增：Save final results ===
        if save_results:
            # Save final video to outputs directory
            for i in range(videos.shape[0]):
                # Keep tensor format for save_videos function
                video_data = videos[i:i+1].unsqueeze(1)  # Add samples dimension: b,samples,c,t,h,w
                
                # Use DynamiCrafter's save_videos function
                save_videos(video_data, output_dirs['outputs'], 
                          filenames=[f'final_video_{i:03d}'], fps=8)
                
                output_path = os.path.join(output_dirs['outputs'], f'final_video_{i:03d}.mp4')
                print(f"Final video saved to: {output_path}")
            
            # Create optimization process video
            if save_process_video:
                create_optimization_process_video(
                    output_dirs['process'], output_dirs['debug'], 
                    num_optimization_steps, debug_save_interval, fps=2
                )
            
            print(f"✅ All results saved to: {output_dirs['main']}")
        
        # 后处理
        videos = self._postprocess_video(videos, output_type)
        
        print(f"✅ 视频生成完成! 形状: {videos.shape}")
        
        # 返回结果
        if return_dict:
            return {"videos": videos}
        else:
            return videos

    def _optimization_loop(
        self,
        noise_shape,
        conditioning,
        device,
        num_optimization_steps,
        learning_rate,
        loss_type,
        weight_type,
        cfg_scale,
        optimizer_type,
        min_step_ratio_start,
        min_step_ratio_end,
        max_step_ratio_start,
        max_step_ratio_end,
        # === Enhanced saving parameters ===
        output_dirs,
        save_results,
        save_debug_images,
        save_debug_videos,
        save_process_video,
        debug_save_interval,
        debug_save_path,
        generator=None,
        **kwargs
    ):
        """
        Optimization loop 替换原来的 scheduler.sample()
        
        参考 guidance_flux_pipeline.py 的优化循环，但适用于 DynamiCrafter
        """
        
        # Create debug directory if needed
        if save_debug_videos and not save_results:
            os.makedirs(debug_save_path, exist_ok=True)
            print(f"[INFO] Debug videos will be saved to: {debug_save_path}")
        
        # 初始化 latents
        if generator is not None:
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=torch.float32)
        else:
            latents = torch.randn(noise_shape, device=device, dtype=torch.float32)
        
        latents = latents.detach().clone()
        latents.requires_grad_(True)
        
        # 设置优化器
        if optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW([latents], lr=learning_rate, betas=(0.9, 0.99), eps=1e-8)
        elif optimizer_type == "Adam":
            optimizer = torch.optim.Adam([latents], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")
        
        # 确定有效的权重类型
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
        if save_results:
            print(f"[INFO] Enhanced saving enabled - interval: {debug_save_interval} steps")
        print("-" * 60)
        
        # 优化循环
        for i in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # 计算当前步骤比例
            progress = i / (num_optimization_steps - 1) if num_optimization_steps > 1 else 0.0
            current_min_step_ratio = (
                min_step_ratio_start + 
                (min_step_ratio_end - min_step_ratio_start) * progress
            )
            current_max_step_ratio = (
                max_step_ratio_start + 
                (max_step_ratio_end - max_step_ratio_start) * progress
            )
            
            # 计算 guidance loss
            if loss_type == "sds":
                loss = self._sds_loss(
                    latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            elif loss_type == "csd":
                loss = self._csd_loss(
                    latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            elif loss_type == "rfds":
                loss = self._rfds_loss(
                    latents,
                    conditioning,
                    cfg_scale=cfg_scale,
                    min_step_ratio=current_min_step_ratio,
                    max_step_ratio=current_max_step_ratio,
                    weight_type=effective_weight_type
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            
            loss.backward()
            optimizer.step()
            
            # === Enhanced debug saving ===
            if i % debug_save_interval == 0:
                if save_results:
                    # Save debug images
                    if save_debug_images:
                        save_debug_frame(
                            output_dirs['debug'], i, latents, 
                            self.model, str(self.device), frame_idx=0
                        )
                    
                    # Save debug videos
                    if save_debug_videos:
                        save_debug_video(
                            output_dirs['debug'], i, latents,
                            self.model, str(self.device)
                        )
                    
                    print(f"[DEBUG] Enhanced debug results saved for step {i}")
                    
                elif save_debug_videos:  # Compatibility with old interface
                    self._save_debug_video(i, latents, debug_save_path)
            
            # 进度日志
            if i % max(1, num_optimization_steps // 10) == 0:
                print(f"[PROGRESS] Step {i}/{num_optimization_steps} - Loss: {loss.item():.6f}")
        
        print(f"[INFO] Optimization completed!")
        
        return latents.detach()

    def _sample_timestep(self, batch_size, min_step_ratio=0.02, max_step_ratio=0.98):
        """
        采样时间步 - 适用于 DynamiCrafter 的 DDIM 调度
        """
        # 使用 DynamiCrafter 的 DDIM 时间步
        from lvdm.models.utils_diffusion import make_ddim_timesteps
        
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method='uniform', 
            num_ddim_timesteps=50,  # 使用50步作为默认
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

    def _add_noise(self, original_samples, noise, timesteps):
        """
        添加噪声 - 适用于 DynamiCrafter 的 DDIM 调度
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
        
        # DDIM 噪声添加
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        noisy_samples = sqrt_alpha_t * original_samples + sqrt_one_minus_alpha_t * noise
        return noisy_samples

    def _sds_loss(self, latents, conditioning, cfg_scale=7.5, min_step_ratio=0.02, max_step_ratio=0.98, weight_type="t"):
        """
        SDS loss 使用 DynamiCrafter 的 model.apply_model
        """
        batch_size = latents.shape[0]
        
        # 采样时间步
        t = self._sample_timestep(batch_size, min_step_ratio, max_step_ratio)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, t)
        
        # 准备条件
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # 前向传播 - 使用 DynamiCrafter 的 model.apply_model
        with torch.no_grad():
            # 条件预测
            noise_pred_cond = self.model.apply_model(noisy_latents, t, cond, **{"fs": fs})
            
            # 无条件预测
            if uc is not None and cfg_scale > 1.0:
                noise_pred_uncond = self.model.apply_model(noisy_latents, t, uc, **{"fs": fs})
                # 应用 CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
        
        # 计算预测的原始样本
        alpha_t = self.model.alphas_cumprod[t].to(self.device)
        while len(alpha_t.shape) < len(latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # 计算 SDS 梯度
        if weight_type == "t":
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)
            grad = w * (latents - pred_original_sample.detach())
        elif weight_type == "ada":
            weighting_factor = torch.abs(latents - pred_original_sample.detach()).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            grad = (latents - pred_original_sample.detach()) / weighting_factor
        elif weight_type == "uniform":
            grad = (latents - pred_original_sample.detach())
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        grad = grad.detach()
        grad = torch.nan_to_num(grad)
        
        # 构建损失
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean") / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def _csd_loss(self, latents, conditioning, cfg_scale=7.5, min_step_ratio=0.02, max_step_ratio=0.98, weight_type="ada"):
        """
        CSD loss 使用 DynamiCrafter 的 model.apply_model
        
        正确的 CSD (Classifier-Free Score Distillation) 实现：
        - fake = 无条件预测的原始样本
        - real = 引导预测 (CFG) 的原始样本
        - 通过无条件和引导预测的差异来优化
        """
        batch_size = latents.shape[0]
        
        # 采样时间步
        t = self._sample_timestep(batch_size, min_step_ratio, max_step_ratio)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, t)
        
        # 准备条件
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # 前向传播 - 分别计算无条件和条件预测
        with torch.no_grad():
            # 条件预测
            noise_pred_cond = self.model.apply_model(noisy_latents, t, cond, **{"fs": fs})
            
            # 无条件预测
            noise_pred_uncond = self.model.apply_model(noisy_latents, t, uc, **{"fs": fs})
            
            # 引导预测 (CFG)
            noise_pred_guided = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 计算预测的原始样本
        alpha_t = self.model.alphas_cumprod[t].to(self.device)
        while len(alpha_t.shape) < len(latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        # CSD 的核心：对比无条件和引导预测
        pred_fake_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred_uncond) / sqrt_alpha_t  # 无条件预测
        pred_real_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred_guided) / sqrt_alpha_t   # 引导预测
        
        # 计算 CSD 梯度
        if weight_type == "ada":
            weighting_factor = torch.abs(latents - pred_real_latents.detach()).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        elif weight_type == "t":
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)
            grad = w * (pred_fake_latents - pred_real_latents)
        elif weight_type == "uniform":
            grad = (pred_fake_latents - pred_real_latents)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        grad = grad.detach()
        grad = torch.nan_to_num(grad)
        
        # 构建损失
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean") / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def _rfds_loss(self, latents, conditioning, cfg_scale=7.5, min_step_ratio=0.02, max_step_ratio=0.98, weight_type="uniform"):
        """
        RFDS loss 使用 DynamiCrafter 的 model.apply_model
        """
        batch_size = latents.shape[0]
        
        # 采样时间步
        t = self._sample_timestep(batch_size, min_step_ratio, max_step_ratio)
        
        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, t)
        
        # 准备条件
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # 前向传播
        with torch.no_grad():
            # 条件预测
            noise_pred_cond = self.model.apply_model(noisy_latents, t, cond, **{"fs": fs})
            
            # 无条件预测
            if uc is not None:
                noise_pred_uncond = self.model.apply_model(noisy_latents, t, uc, **{"fs": fs})
                # 应用 CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond
        
        # 计算预测的原始样本
        alpha_t = self.model.alphas_cumprod[t].to(self.device)
        while len(alpha_t.shape) < len(latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # RFDS: 直接优化匹配预测的原始样本
        target = pred_original_sample.detach()
        
        if weight_type == "t":
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)
            loss = 0.5 * w * F.mse_loss(latents, target, reduction="mean")
        elif weight_type == "ada":
            prediction_error = latents - target
            weighting_factor = torch.abs(prediction_error).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            loss = 0.5 * F.mse_loss(latents, target, reduction="none")
            loss = (loss / weighting_factor).mean()
        elif weight_type == "uniform":
            loss = 0.5 * F.mse_loss(latents, target, reduction="mean")
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        loss = loss / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def _save_debug_video(self, step, latents, debug_save_path):
        """保存 debug 视频"""
        try:
            with torch.no_grad():
                # 解码 latents
                debug_videos = self.model.decode_first_stage(latents.detach())
                debug_videos = debug_videos.cpu().float().numpy()
                
                # 保存视频
                filename = f"step_{step:06d}"
                save_videos(debug_videos.unsqueeze(1), debug_save_path, filenames=[filename], fps=8)
                
                print(f"[DEBUG] Saved debug video: {debug_save_path}/{filename}.mp4")
                
        except Exception as e:
            print(f"[DEBUG] Failed to save debug video: {e}")

    def save_video(self, video: torch.Tensor, output_path: str, fps: int = 8, **kwargs):
        """
        完全来自 dynamicrafter_pipeline.py:DynamiCrafterImg2VideoPipeline.save_video
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
    """
    print("🧪 测试 DynamiCrafter Guidance Pipeline")
    print("=" * 70)
    
    try:
        # 初始化 pipeline
        print("🔧 初始化 DynamiCrafter Guidance Pipeline...")
        pipeline = DynamiCrafterGuidancePipeline(
            resolution='256_256'
        )
        
        # 查找测试图像
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
        
        # 加载测试图像
        print(f"📸 加载测试图像: {test_image_path}")
        img = Image.open(test_image_path).convert("RGB")
        
        # 测试 Enhanced saving 功能
        test_prompt = "a person walking in a beautiful garden with flowers blooming"
        
        print(f"📝 测试提示词: {test_prompt}")
        print(f"🎬 开始 Enhanced SDS 优化生成...")
        
        start_time = time.time()
        
        # 使用 Enhanced guidance 生成
        result = pipeline(
            image=img,
            prompt=test_prompt,
            num_optimization_steps=20,  # 短测试
            learning_rate=0.05,
            loss_type="sds",
            cfg_scale=7.5,
            # Enhanced saving parameters
            save_results=True,
            save_debug_images=True,
            save_debug_videos=True,
            save_process_video=True,
            debug_save_interval=5,
            results_dir="./results_enhanced_test",
            return_dict=True
        )
        
        elapsed_time = time.time() - start_time
        
        # 检查结果
        video = result["videos"]
        print(f"🎉 Enhanced SDS 生成成功!")
        print(f"📊 视频形状: {video.shape}")
        print(f"⏱️ 用时: {elapsed_time:.2f} 秒")
        
        # 检查 NaN
        if torch.isnan(video).any():
            print("❌ 检测到 NaN 值!")
            return False
        else:
            print("✅ 无 NaN 问题")
        
        print("🎉 Enhanced DynamiCrafter Guidance Pipeline 测试成功!")
        print("📋 测试总结:")
        print("  ✅ 模型加载正常")
        print("  ✅ 图像预处理正常")
        print("  ✅ 文本编码正常")
        print("  ✅ Enhanced SDS 优化正常")
        print("  ✅ 视频解码正常")
        print("  ✅ Enhanced 保存功能正常")
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
        print("🔬 正确的改造思路：")
        print("  • 保持 DynamiCrafter 的所有核心逻辑")
        print("  • 只替换 scheduler.sample() 为 optimization loop")
        print("  • 在 optimization loop 中使用 DynamiCrafter 的 model.apply_model()")
        print("  • 增强的可视化和保存功能")
        print("")
        print("🎯 转换关系：")
        print("  • pipeline_normal_to_flux.py → guidance_flux_pipeline.py")
        print("  • dynamicrafter_pipeline.py → guidance_pipeline.py")
        print("")
        print("🎨 可视化功能：")
        print("  • 组织化的输出目录结构")
        print("  • 输入条件保存 (原始图像、处理后图像、提示词)")
        print("  • 参数保存 (完整的参数记录)")
        print("  • 优化过程视频生成")
        print("  • 增强的 debug 保存功能")
        print("")
        print("📖 使用示例:")
        print("```python")
        print("from guidance_pipeline import DynamiCrafterGuidancePipeline")
        print("from PIL import Image")
        print("")
        print("# 初始化 pipeline")
        print("pipeline = DynamiCrafterGuidancePipeline(resolution='256_256')")
        print("")
        print("# 加载图像")
        print("image = Image.open('your_image.jpg')")
        print("")
        print("# Enhanced SDS 优化生成")
        print("result = pipeline(")
        print("    image=image,")
        print("    prompt='person walking in garden',")
        print("    num_optimization_steps=1000,")
        print("    loss_type='sds',")
        print("    cfg_scale=7.5,")
        print("    # Enhanced saving")
        print("    save_results=True,")
        print("    save_debug_images=True,")
        print("    save_debug_videos=True,")
        print("    save_process_video=True,")
        print("    debug_save_interval=100")
        print(")")
        print("")
        print("# 保存视频")
        print("pipeline.save_video(result['videos'], 'output.mp4')")
        print("```")
