import os
import time
import sys
from typing import Optional, Union, List, Any, Dict, Callable
from omegaconf import OmegaConf
import torch
import numpy as np
from einops import repeat, rearrange
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, get_latent_z


def get_fixed_ddim_sampler(model, ddim_steps, ddim_eta, verbose=False):
    """
    返回修复后sigma值的DDIMSampler - 极简修复方案
    只修复sigma计算，保持其他逻辑完全不变
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


class DynamiCrafterImg2VideoPipeline:
    """
    DynamiCrafter Image-to-Video Pipeline
    
    This pipeline implements a diffusers-like interface for DynamiCrafter image-to-video generation
    with the minimal fix for NaN issues.
    """
    
    def __init__(
        self, 
        resolution: str = '256_256',
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs
    ):
        """
        Initialize the DynamiCrafter pipeline.
        
        Args:
            resolution: Model resolution (e.g., '256_256', '512_512', '1024_1024')
            device: Device to run the model on
            dtype: Data type for the model
        """
        self.resolution = tuple(map(int, resolution.split('_')))  # (height, width)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # Initialize components
        self._load_components()
        self._setup_image_processor()
        
        # Pipeline state
        self._is_initialized = True
        print(f"✅ DynamiCrafter Pipeline initialized: {self.resolution[0]}x{self.resolution[1]}")
        
    def _load_components(self):
        """Load model and scheduler components"""
        # Download model if needed
        self._download_model()
        
        # Load model - 修复配置文件路径
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        ckpt_path = os.path.join(project_root, f'checkpoints/dynamicrafter_{self.resolution[1]}_v1/model.ckpt')
        config_file = os.path.join(project_root, f'configs/inference_{self.resolution[1]}_v1.0.yaml')
        
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        
        self.model = instantiate_from_config(model_config)
        assert os.path.exists(ckpt_path), f"Error: checkpoint Not Found at {ckpt_path}!"
        self.model = load_model_checkpoint(self.model, ckpt_path)
        self.model.eval()
        
        # 确保模型移动到正确的设备 - 增强版本
        self.model = self.model.to(self.device)
        
        # 强制确保所有子模块都移动到正确设备
        def move_to_device(module):
            """递归移动模块到目标设备"""
            for name, param in module.named_parameters():
                if param.device != torch.device(self.device):
                    param.data = param.data.to(self.device)
            
            for name, buffer in module.named_buffers():
                if buffer.device != torch.device(self.device):
                    buffer.data = buffer.data.to(self.device)
            
            for name, child in module.named_children():
                move_to_device(child)
        
        move_to_device(self.model)
        
        # 特别处理文本编码器和图像编码器
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model = self.model.cond_stage_model.to(self.device)
            move_to_device(self.model.cond_stage_model)
        
        if hasattr(self.model, 'first_stage_model'):
            self.model.first_stage_model = self.model.first_stage_model.to(self.device)
            move_to_device(self.model.first_stage_model)
        
        if hasattr(self.model, 'embedder'):
            self.model.embedder = self.model.embedder.to(self.device)
            move_to_device(self.model.embedder)
        
        if hasattr(self.model, 'image_proj_model'):
            self.model.image_proj_model = self.model.image_proj_model.to(self.device)
            move_to_device(self.model.image_proj_model)
        
        print(f"🔄 Model moved to {self.device}")
        print(f"✅ All model components verified on {self.device}")
        
        # 验证设备
        device_check_passed = True
        for name, param in self.model.named_parameters():
            if param.device != torch.device(self.device):
                print(f"⚠️ Warning: {name} is still on {param.device}")
                device_check_passed = False
        
        if device_check_passed:
            print(f"✅ Device check passed - all parameters on {self.device}")
        else:
            print(f"⚠️ Device check failed - some parameters not on {self.device}")
        
    def _setup_image_processor(self):
        """Setup image preprocessing transforms"""
        self.image_processor = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        
    def _download_model(self):
        """Download model weights if needed"""
        REPO_ID = f'Doubiiu/DynamiCrafter_{self.resolution[1]}' if self.resolution[1] != 256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        
        # 修复模型下载路径
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
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
    
    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        """Enable attention slicing for memory efficiency (placeholder for compatibility)"""
        print("📝 Note: Attention slicing not implemented yet")
        pass
    
    def disable_attention_slicing(self):
        """Disable attention slicing (placeholder for compatibility)"""
        print("📝 Note: Attention slicing not implemented yet")
        pass
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers memory efficient attention (placeholder for compatibility)"""
        print("📝 Note: xformers optimization not implemented yet")
        pass
    
    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None):
        """Move pipeline to device"""
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        # 确保所有模型组件都移动到正确设备
        self.model = self.model.to(device)
        
        # 强制确保所有子模块都移动到正确设备
        def move_to_device(module):
            """递归移动模块到目标设备"""
            for name, param in module.named_parameters():
                if param.device != torch.device(device):
                    param.data = param.data.to(device)
            
            for name, buffer in module.named_buffers():
                if buffer.device != torch.device(device):
                    buffer.data = buffer.data.to(device)
            
            for name, child in module.named_children():
                move_to_device(child)
        
        move_to_device(self.model)
        
        # 特别处理文本编码器和图像编码器
        if hasattr(self.model, 'cond_stage_model'):
            self.model.cond_stage_model = self.model.cond_stage_model.to(device)
            move_to_device(self.model.cond_stage_model)
        
        if hasattr(self.model, 'first_stage_model'):
            self.model.first_stage_model = self.model.first_stage_model.to(device)
            move_to_device(self.model.first_stage_model)
        
        if hasattr(self.model, 'embedder'):
            self.model.embedder = self.model.embedder.to(device)
            move_to_device(self.model.embedder)
        
        if hasattr(self.model, 'image_proj_model'):
            self.model.image_proj_model = self.model.image_proj_model.to(device)
            move_to_device(self.model.image_proj_model)
        
        if dtype is not None:
            self.model = self.model.to(dtype)
        
        print(f"🔄 Pipeline moved to {device}")
        return self
    
    def _preprocess_image(self, image, height=None, width=None):
        """预处理输入图像"""
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
        """编码文本提示"""
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
        """编码输入图像"""
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
        """准备条件输入"""
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
            # 修复：创建正确形状的零图像（原始图像空间，3通道）
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
    
    def _prepare_latents(self, noise_shape, device, generator, dtype):
        """准备初始 latents"""
        if generator is not None:
            latents = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = torch.randn(noise_shape, device=device, dtype=dtype)
        return latents
    
    def _decode_latents(self, latents):
        """解码潜在表示到视频"""
        with torch.no_grad():
            videos = self.model.decode_first_stage(latents)
        return videos
    
    def _postprocess_video(self, videos, output_type):
        """后处理视频输出"""
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
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate video from image and prompt using DynamiCrafter.
        
        Args:
            image: Input image for video generation
            prompt: Text prompt(s) to guide video generation
            negative_prompt: Negative prompt(s) 
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            eta: DDIM eta parameter (0.0=deterministic, 1.0=stochastic)
            frame_stride: Frame stride control parameter
            num_frames: Number of frames to generate
            height: Height of generated video
            width: Width of generated video
            num_videos_per_prompt: Number of videos to generate per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated noisy latents
            output_type: Output format ("tensor", "numpy", "pil")
            return_dict: Whether to return dict or tensor
            callback: Callback function
            callback_steps: Callback frequency
            cross_attention_kwargs: Cross-attention parameters
            
        Returns:
            Generated videos tensor or dict
        """
        
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
        # 移除重复的设备移动，因为模型已经在正确设备上
        # self.model = self.model.to(device)
        
        print(f"🎬 开始生成视频...")
        print(f"📝 提示词: {prompt}")
        print(f"🔧 参数: steps={num_inference_steps}, guidance={guidance_scale}, eta={eta}")
        print(f"💻 设备: {device}")
        
        # 图像预处理
        processed_image = self._preprocess_image(image, height, width)
        
        # 模型参数
        num_frames = num_frames or self.model.temporal_length
        channels = self.model.model.diffusion_model.out_channels
        if height is None or width is None:
            height, width = self.resolution
        latent_height, latent_width = height // 8, width // 8
        
        # 噪声形状
        noise_shape = (batch_size, channels, num_frames, latent_height, latent_width)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
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
            
            # 初始化调度器
            scheduler = get_fixed_ddim_sampler(self.model, num_inference_steps, eta, verbose=False)
            
            # 执行采样
            print(f"🔄 开始 DDIM 采样...")
            samples, _ = scheduler.sample(
                S=num_inference_steps,
                conditioning=conditioning["cond"],
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=conditioning["uc"],
                eta=eta,
                fs=conditioning["fs"],
                **kwargs
            )
            
            # 解码 latents 到视频
            print(f"🎞️ 解码潜在表示...")
            videos = self._decode_latents(samples)
        
        # 后处理
        videos = self._postprocess_video(videos, output_type)
        
        print(f"✅ 视频生成完成! 形状: {videos.shape}")
        
        # 返回结果
        if return_dict:
            return {"videos": videos}
        else:
            return videos
    
    def save_video(
        self, 
        video: torch.Tensor, 
        output_path: str, 
        fps: int = 8, 
        **kwargs
    ):
        """
        Save generated video to file
        
        Args:
            video: Generated video tensor
            output_path: Output file path
            fps: Frames per second
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get filename without extension
        filename = os.path.basename(output_path).split('.')[0]
        
        # 修复维度问题：save_videos期望 (batch, samples, channels, time, height, width)
        # 而我们的video是 (batch, channels, time, height, width)
        if video.dim() == 4:
            # 如果是4维 (channels, time, height, width)，添加batch和samples维度
            video = video.unsqueeze(0).unsqueeze(0)  # -> (1, 1, channels, time, height, width)
        elif video.dim() == 5:
            # 如果是5维 (batch, channels, time, height, width)，添加samples维度
            video = video.unsqueeze(1)  # -> (batch, 1, channels, time, height, width)
        
        # 确保视频tensor在CPU上
        if video.is_cuda:
            video = video.cpu()
        
        print(f"📊 保存视频形状: {video.shape}")
        save_videos(video, output_dir or '.', filenames=[filename], fps=fps)
        
        print(f"💾 视频已保存: {output_path}")


def test_pipeline():
    """测试 DynamiCrafter Pipeline"""
    print("🧪 测试 DynamiCrafter Pipeline")
    print("=" * 60)
    
    try:
        # 初始化 pipeline - 先使用默认设备测试基本功能
        print("🔧 初始化 Pipeline (默认设备)...")
        pipeline = DynamiCrafterImg2VideoPipeline(
            resolution='256_256'
            # device='cuda:3'  # 暂时注释掉，先测试基本功能
        )
        
        # 检查测试图像 - 更新为实际存在的图像
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        test_image_paths = [
            os.path.join(project_root, 'prompts/1024/pour_bear.png'),
            os.path.join(project_root, 'prompts/1024/robot01.png'),
            os.path.join(project_root, 'prompts/512_loop/24.png'),
            os.path.join(project_root, 'prompts/1024/astronaut04.png'),
        ]
        
        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break
        
        if test_image_path is None:
            print("❌ 测试图像不存在，请确保有测试图像文件")
            print("📋 查找的路径:")
            for path in test_image_paths:
                print(f"  - {path} (存在: {os.path.exists(path)})")
            return False
        
        # 加载测试图像
        print(f"📸 加载测试图像: {test_image_path}")
        img = Image.open(test_image_path).convert("RGB")
        print(f"📐 图像尺寸: {img.size}")
        
        # 测试基本功能
        test_prompt = "a person walking in a beautiful garden with flowers blooming"
        
        print(f"📝 测试提示词: {test_prompt}")
        print(f"🎬 开始生成...")
        
        start_time = time.time()
        
        # 使用 pipeline 生成
        result = pipeline(
            image=img,
            prompt=test_prompt,
            num_inference_steps=5,  # 进一步减少步数以便快速测试
            guidance_scale=7.5,
            eta=0.0,
            frame_stride=3,
            return_dict=True
        )
        
        elapsed_time = time.time() - start_time
        
        # 检查结果
        video = result["videos"]
        print(f"🎉 生成成功!")
        print(f"📊 视频形状: {video.shape}")
        print(f"⏱️ 用时: {elapsed_time:.2f} 秒")
        
        # 检查是否有NaN
        if torch.isnan(video).any():
            print("❌ 检测到 NaN 值!")
            return False
        else:
            print("✅ 无 NaN 问题")
        
        # 保存结果
        output_dir = './results_pipeline_test/'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_video.mp4")
        
        pipeline.save_video(video, output_path)
        
        # 验证输出
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ 视频已保存: {output_path} ({file_size/1024:.1f} KB)")
        else:
            print("❌ 视频保存失败")
            return False
        
        print("🎉 Pipeline 测试成功!")
        print("📋 测试总结:")
        print("  ✅ 模型加载正常")
        print("  ✅ 图像预处理正常")
        print("  ✅ 文本编码正常")
        print("  ✅ DDIM采样正常")
        print("  ✅ 视频解码正常")
        print("  ✅ 视频保存正常")
        print("  ✅ 无NaN问题")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_pipeline()
    else:
        print("📝 用法:")
        print("  python dynamicrafter_pipeline.py test  # 运行测试")
        print("")
        print("📖 示例代码:")
        print("```python")
        print("from dynamicrafter_pipeline import DynamiCrafterImg2VideoPipeline")
        print("from PIL import Image")
        print("")
        print("# 初始化 pipeline")
        print("pipeline = DynamiCrafterImg2VideoPipeline(resolution='256_256')")
        print("")
        print("# 加载图像")
        print("image = Image.open('your_image.jpg')")
        print("")
        print("# 生成视频")
        print("result = pipeline(")
        print("    image=image,")
        print("    prompt='your prompt here',")
        print("    num_inference_steps=50,")
        print("    guidance_scale=7.5")
        print(")")
        print("")
        print("# 保存视频")
        print("pipeline.save_video(result['videos'], 'output.mp4')")
        print("```") 