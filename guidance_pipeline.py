# guidance_pipeline.py
"""
DynamiCrafter Guidance Pipeline - Simplified Version with Complete Debug

专注于核心的 SDS loss 逻辑：
1. 保持 DynamiCrafter 的所有核心逻辑
2. 只替换 scheduler.sample() 调用为 optimization loop
3. 在 optimization loop 中使用 DynamiCrafter 的 model.apply_model() 方法
4. 完整的debug保存功能（类似原版结构）
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Union, List, Dict, Any
from omegaconf import OmegaConf
# 添加debug相关导入
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, get_latent_z


class DynamiCrafterGuidancePipeline:
    """
    DynamiCrafter Guidance Pipeline - Simplified Version with Complete Debug
    专注于核心的 SDS loss 逻辑 + 完整debug功能
    """
    
    def __init__(self, resolution: str = '256_256', device: Optional[str] = None, debug_dir: Optional[str] = None):
        self.resolution = tuple(map(int, resolution.split('_')))
        
        # Debug设置
        self.debug_enabled = debug_dir is not None
        if self.debug_enabled:
            # 使用类似原版的目录命名：timestamp_prompt_...（稍后设置）
            self.debug_base_dir = debug_dir
            self.debug_dir = None  # 稍后根据prompt创建具体目录
            print(f"🐛 Debug mode enabled, base dir: {debug_dir}")
        else:
            self.debug_dir = None
        
        # Enhanced device selection
        if device is None:
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if cuda_visible is not None:
                device_num = cuda_visible.split(',')[0]
                self.device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
            else:
                if torch.cuda.is_available():
                    self.device = "cuda:3" if torch.cuda.device_count() > 3 else "cuda:0"
                else:
                    self.device = "cpu"
        else:
            self.device = device
        
        print(f"🔧 Device: {self.device}, Resolution: {self.resolution}")
        
        # Load model
        self._load_components()
        self._setup_image_processor()
        
        print(f"✅ DynamiCrafter Guidance Pipeline initialized")
        
    def _load_components(self):
        """加载 DynamiCrafter 模型"""
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
        
        # 设置 perframe_ae
        if self.resolution[1] in [512, 1024]:
            self.model.perframe_ae = True
            print(f"✅ Set perframe_ae=True for {self.resolution[1]} model")
        else:
            self.model.perframe_ae = False
            print(f"✅ Set perframe_ae=False for {self.resolution[1]} model")
        
        # Move to device
        self.model = self.model.to(self.device)
        self._ensure_device_consistency()
        
        print(f"✅ Model loaded: {self.resolution[1]}")
        
    def _ensure_device_consistency(self):
        """确保设备一致性"""
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
        """设置图像处理器"""
        self.image_processor = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        
    def _download_model(self):
        """下载模型"""
        from huggingface_hub import hf_hub_download
        
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
        """预处理图像"""
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
        """编码提示词"""
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
        """编码图像"""
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
                             frame_stride, guidance_scale, batch_size, num_frames=None):
        """准备条件输入"""
        # 组合文本和图像嵌入
        cond_emb = torch.cat([text_embeddings["cond"], image_embeddings], dim=1)
        cond = {"c_crossattn": [cond_emb]}
        
        # 图像条件（hybrid模式）
        if self.model.model.conditioning_key == 'hybrid':
            img_cat_cond = image_latents[:,:,:1,:,:]
            # 使用实际的 num_frames
            actual_num_frames = num_frames if num_frames is not None else self.model.temporal_length
            img_cat_cond = img_cat_cond.repeat(1, 1, actual_num_frames, 1, 1)
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
        """解码潜在表示"""
        with torch.no_grad():
            videos = self.model.decode_first_stage(latents)
        return videos

    def _sample_timestep(self, batch_size, min_step_ratio=0.02, max_step_ratio=0.98):
        """采样时间步"""
        from lvdm.models.utils_diffusion import make_ddim_timesteps
        
        # 根据模型分辨率选择合适的 timestep_spacing
        if self.resolution[1] in [512, 1024]:
            timestep_spacing = 'uniform_trailing'
        else:
            timestep_spacing = 'uniform'
        
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=timestep_spacing,
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
        """添加噪声 - DDIM 调度"""
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

    def _apply_guidance_rescale(self, noise_pred_cond, noise_pred_uncond, cfg_scale):
        """应用 guidance rescale"""
        # 标准 CFG
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # 对于 512 和 1024 模型，应用 guidance_rescale
        if self.resolution[1] in [512, 1024]:
            guidance_rescale = 0.7
            
            # 实现 guidance rescale 逻辑
            std_text = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
            std_cfg = noise_pred.std(dim=list(range(1, noise_pred.ndim)), keepdim=True)
            
            # rescale the results from guidance (prevent over-exposure)
            noise_pred_rescaled = noise_pred * (std_text / std_cfg)
            
            # mix with the original results from guidance by factor guidance_rescale
            noise_pred = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_pred
        
        return noise_pred

    def _sds_loss(self, latents, conditioning, cfg_scale=7.5, min_step_ratio=0.02, max_step_ratio=0.98, weight_type="t"):
        """
        核心 SDS loss 计算
        
        SDS (Score Distillation Sampling) Loss 原理：
        1. 随机采样时间步 t
        2. 向 latents 添加噪声得到 noisy_latents
        3. 使用预训练模型预测噪声 noise_pred
        4. 通过 DDIM reverse 计算预测的原始样本
        5. 计算 SDS 梯度：∇_latents = w(t) * (latents - pred_original_sample)
        6. 构建巧妙的损失函数，使其梯度等于 SDS 梯度
        """
        batch_size = latents.shape[0]
        
        # 1. 采样时间步
        t = self._sample_timestep(batch_size, min_step_ratio, max_step_ratio)
        
        # 2. 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self._add_noise(latents, noise, t)
        
        # 3. 准备条件
        cond = conditioning["cond"]
        uc = conditioning["uc"]
        fs = conditioning["fs"]
        
        # 4. 前向传播 - 使用 DynamiCrafter 的 model.apply_model
        with torch.no_grad():
            # 条件预测
            noise_pred_cond = self.model.apply_model(noisy_latents, t, cond, **{"fs": fs})
            
            # 无条件预测
            if uc is not None and cfg_scale > 1.0:
                noise_pred_uncond = self.model.apply_model(noisy_latents, t, uc, **{"fs": fs})
                # 应用 guidance_rescale (对 512/1024 模型关键)
                noise_pred = self._apply_guidance_rescale(noise_pred_cond, noise_pred_uncond, cfg_scale)
            else:
                noise_pred = noise_pred_cond
        
        # 5. 计算预测的原始样本 (DDIM reverse)
        alpha_t = self.model.alphas_cumprod[t].to(self.device)
        while len(alpha_t.shape) < len(latents.shape):
            alpha_t = alpha_t.unsqueeze(-1)
        
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
        
        # DDIM reverse formula: x_0 = (x_t - sqrt(1-α_t) * ε_θ) / sqrt(α_t)
        pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        # 6. 计算 SDS 梯度 (不同权重策略)
        if weight_type == "t":
            # 基于时间步的权重 w(t) = 1 - α_t
            w = (1.0 - alpha_t).view(batch_size, 1, 1, 1, 1)
            grad = w * (latents - pred_original_sample.detach())
        elif weight_type == "ada":
            # 自适应权重：基于预测误差的大小
            weighting_factor = torch.abs(latents - pred_original_sample.detach()).mean(dim=(1, 2, 3, 4), keepdim=True)
            weighting_factor = torch.clamp(weighting_factor, 1e-4)
            grad = (latents - pred_original_sample.detach()) / weighting_factor
        elif weight_type == "uniform":
            # 均匀权重
            grad = (latents - pred_original_sample.detach())
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        grad = grad.detach()
        grad = torch.nan_to_num(grad)
        
        # 7. 构建巧妙的损失函数
        # 关键技巧：构造损失函数 L = 0.5 * ||latents - target||^2
        # 其中 target = latents - grad，使得 ∇_latents L = latents - target = grad
        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="mean") / batch_size
        
        torch.cuda.empty_cache()
        
        return loss

    def _create_debug_structure(self, prompt: str, config_info: Dict):
        """创建完整的debug目录结构"""
        if not self.debug_enabled:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 清理prompt用于文件名
            safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in prompt)
            safe_prompt = '_'.join(safe_prompt.split())[:50]  # 限制长度
            
            # 创建类似原版的目录名
            loss_type = config_info.get('loss_type', 'sds')
            weight_type = config_info.get('weight_type', 't')
            lr = config_info.get('learning_rate', 0.05)
            steps = config_info.get('num_optimization_steps', 100)
            cfg = config_info.get('cfg_scale', 7.5)
            
            dir_name = f"{timestamp}_{safe_prompt}__{loss_type}_{weight_type}_lr{lr}_steps{steps}_cfg{cfg}"
            self.debug_dir = os.path.join(self.debug_base_dir, dir_name)
            
            # 创建子目录结构
            self.debug_subdirs = {
                'inputs': os.path.join(self.debug_dir, 'inputs'),
                'outputs': os.path.join(self.debug_dir, 'outputs'),
                'process': os.path.join(self.debug_dir, 'process'),
                'debug': os.path.join(self.debug_dir, 'debug'),
                'params': os.path.join(self.debug_dir, 'params')
            }
            
            for subdir in self.debug_subdirs.values():
                os.makedirs(subdir, exist_ok=True)
            
            print(f"🐛 Debug structure created: {self.debug_dir}")
            
        except Exception as e:
            print(f"⚠️  Debug structure creation failed: {e}")

    def _save_input_files(self, original_image, processed_image, prompt: str):
        """保存输入文件到inputs目录"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs'):
            return
        
        try:
            inputs_dir = self.debug_subdirs['inputs']
            
            # 保存原始图像
            if isinstance(original_image, Image.Image):
                original_image.save(os.path.join(inputs_dir, 'original_image.png'))
            elif isinstance(original_image, np.ndarray):
                if original_image.max() <= 1.0:
                    original_image = (original_image * 255).astype(np.uint8)
                Image.fromarray(original_image).save(os.path.join(inputs_dir, 'original_image.png'))
            
            # 保存预处理图像
            if isinstance(processed_image, torch.Tensor):
                # 转换为PIL格式保存
                if processed_image.dim() == 3:
                    # 反标准化: [-1, 1] -> [0, 1]
                    processed_np = ((processed_image + 1.0) / 2.0).clamp(0, 1)
                    processed_np = processed_np.permute(1, 2, 0).cpu().numpy()
                    processed_np = (processed_np * 255).astype(np.uint8)
                    Image.fromarray(processed_np).save(os.path.join(inputs_dir, 'processed_image.png'))
            
            # 保存prompt
            with open(os.path.join(inputs_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            print(f"✅ Input files saved to {inputs_dir}")
            
        except Exception as e:
            print(f"⚠️  Input files save failed: {e}")

    def _save_parameters(self, config_info: Dict):
        """保存参数配置到params目录"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs'):
            return
        
        try:
            params_dir = self.debug_subdirs['params']
            
            # 保存为txt格式（类似原版）
            txt_path = os.path.join(params_dir, 'parameters.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("DynamiCrafter Guidance Pipeline Parameters\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for key, value in config_info.items():
                    f.write(f"{key}: {value}\n")
            
            # 保存为json格式
            json_path = os.path.join(params_dir, 'parameters.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Parameters saved to {params_dir}")
            
        except Exception as e:
            print(f"⚠️  Parameters save failed: {e}")

    def _save_debug_step(self, step: int, loss: float, latents: torch.Tensor, 
                        conditioning: Dict = None, save_interval: int = 100):
        """保存debug步骤信息（类似原版每100步保存）"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs'):
            return
        
        # 保存loss到列表（每步都记录）
        if not hasattr(self, 'debug_losses'):
            self.debug_losses = []
        self.debug_losses.append(loss)
        
        # 每save_interval步保存详细debug信息
        if step % save_interval == 0:
            try:
                debug_dir = self.debug_subdirs['debug']
                
                with torch.no_grad():
                    # 解码完整视频
                    videos = self.model.decode_first_stage(latents)
                    videos = torch.clamp((videos + 1.0) / 2.0, 0, 1)
                    
                    batch_size, channels, num_frames, height, width = videos.shape
                    video_np = videos[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
                    
                    # 保存第一帧（类似原版的frame_00.png）
                    first_frame = (video_np[0] * 255).astype(np.uint8)
                    frame_path = os.path.join(debug_dir, f"step_{step:06d}_frame_00.png")
                    Image.fromarray(first_frame).save(frame_path)
                    
                    # 保存中间帧（类似原版的frame.png）
                    mid_frame_idx = num_frames // 2
                    mid_frame = (video_np[mid_frame_idx] * 255).astype(np.uint8)
                    mid_frame_path = os.path.join(debug_dir, f"step_{step:06d}_frame.png")
                    Image.fromarray(mid_frame).save(mid_frame_path)
                    
                    # 保存完整视频（类似原版的video.mp4）
                    video_path = os.path.join(debug_dir, f"step_{step:06d}_video.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 8.0, (width, height))
                    
                    for frame_idx in range(num_frames):
                        frame = (video_np[frame_idx] * 255).astype(np.uint8)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    
                    out.release()
                    
                    print(f"🐛 Debug step {step} saved: frame_00, frame, video")
                    
            except Exception as e:
                print(f"⚠️  Debug step {step} save failed: {e}")

    def _save_process_video(self):
        """创建优化过程视频（类似原版的optimization_process.mp4）"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs'):
            return
        
        try:
            debug_dir = self.debug_subdirs['debug']
            process_dir = self.debug_subdirs['process']
            
            # 收集所有中间帧
            frame_files = []
            for file in sorted(os.listdir(debug_dir)):
                if file.endswith('_frame.png'):  # 使用中间帧
                    frame_files.append(os.path.join(debug_dir, file))
            
            if len(frame_files) < 2:
                print("⚠️  Not enough frames for process video")
                return
            
            # 创建优化过程视频
            process_video_path = os.path.join(process_dir, 'optimization_process.mp4')
            
            # 读取第一帧确定尺寸
            first_frame = cv2.imread(frame_files[0])
            height, width = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(process_video_path, fourcc, 2.0, (width, height))  # 较慢的帧率
            
            for frame_file in frame_files:
                frame = cv2.imread(frame_file)
                out.write(frame)
            
            out.release()
            print(f"✅ Process video saved: {process_video_path}")
            
        except Exception as e:
            print(f"⚠️  Process video creation failed: {e}")

    def _save_final_outputs(self, videos: torch.Tensor):
        """保存最终输出到outputs目录"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs'):
            return
        
        try:
            outputs_dir = self.debug_subdirs['outputs']
            
            # 保存最终视频（类似原版的final_video_000.mp4）
            videos_np = videos.detach().cpu().numpy()
            videos_np = np.clip((videos_np + 1.0) / 2.0, 0, 1)
            
            batch_size, channels, num_frames, height, width = videos_np.shape
            
            for batch_idx in range(batch_size):
                video_np = videos_np[batch_idx].transpose(1, 2, 3, 0)  # [T, H, W, C]
                
                # 保存为MP4（主要格式）
                mp4_path = os.path.join(outputs_dir, f'final_video_{batch_idx:03d}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(mp4_path, fourcc, 8.0, (width, height))
                
                for frame_idx in range(num_frames):
                    frame = (video_np[frame_idx] * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                out.release()
                
                # 额外保存为GIF（便于预览）
                gif_path = os.path.join(outputs_dir, f'final_video_{batch_idx:03d}.gif')
                gif_frames = []
                for frame_idx in range(num_frames):
                    frame = (video_np[frame_idx] * 255).astype(np.uint8)
                    gif_frames.append(Image.fromarray(frame))
                
                gif_frames[0].save(
                    gif_path, 
                    save_all=True, 
                    append_images=gif_frames[1:], 
                    duration=125,  # 8fps
                    loop=0
                )
            
            print(f"✅ Final outputs saved to {outputs_dir}")
            
        except Exception as e:
            print(f"⚠️  Final outputs save failed: {e}")

    def _save_loss_analysis(self):
        """保存loss分析到process目录"""
        if not self.debug_enabled or not hasattr(self, 'debug_subdirs') or not hasattr(self, 'debug_losses'):
            return
        
        try:
            process_dir = self.debug_subdirs['process']
            
            # 保存loss数据
            loss_data_path = os.path.join(process_dir, 'loss_data.txt')
            with open(loss_data_path, 'w') as f:
                f.write("# Step\tLoss\n")
                for i, loss in enumerate(self.debug_losses):
                    f.write(f"{i}\t{loss:.8f}\n")
            
            # 创建loss分析图
            if len(self.debug_losses) > 1:
                plt.figure(figsize=(15, 10))
                
                # 主图：完整loss曲线
                plt.subplot(2, 3, 1)
                plt.plot(self.debug_losses)
                plt.title('Complete SDS Loss Curve')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.grid(True)
                
                # 子图：最后50步
                plt.subplot(2, 3, 2)
                recent_losses = self.debug_losses[-50:] if len(self.debug_losses) > 50 else self.debug_losses
                plt.plot(range(len(self.debug_losses)-len(recent_losses), len(self.debug_losses)), recent_losses)
                plt.title('Last 50 Steps')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True)
                
                # 子图：loss变化率
                loss_diff = np.diff(self.debug_losses)
                plt.subplot(2, 3, 3)
                plt.plot(loss_diff)
                plt.title('Loss Change Rate')
                plt.xlabel('Step')
                plt.ylabel('∆Loss')
                plt.grid(True)
                
                # 子图：loss分布
                plt.subplot(2, 3, 4)
                plt.hist(self.debug_losses, bins=50, alpha=0.7)
                plt.title('Loss Distribution')
                plt.xlabel('Loss Value')
                plt.ylabel('Frequency')
                plt.yscale('log')
                
                # 子图：滑动平均
                if len(self.debug_losses) > 10:
                    window_size = min(10, len(self.debug_losses) // 10)
                    moving_avg = np.convolve(self.debug_losses, np.ones(window_size)/window_size, mode='valid')
                    plt.subplot(2, 3, 5)
                    plt.plot(self.debug_losses, alpha=0.3, label='Original')
                    plt.plot(range(window_size-1, len(self.debug_losses)), moving_avg, label=f'Moving Avg ({window_size})')
                    plt.title('Loss with Moving Average')
                    plt.xlabel('Step')
                    plt.ylabel('Loss')
                    plt.yscale('log')
                    plt.legend()
                    plt.grid(True)
                
                # 子图：loss统计
                plt.subplot(2, 3, 6)
                stats_text = f"""
                Total Steps: {len(self.debug_losses)}
                Final Loss: {self.debug_losses[-1]:.6e}
                Min Loss: {min(self.debug_losses):.6e}
                Max Loss: {max(self.debug_losses):.6e}
                Mean Loss: {np.mean(self.debug_losses):.6e}
                Std Loss: {np.std(self.debug_losses):.6e}
                """
                plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                         fontsize=10, verticalalignment='center', fontfamily='monospace')
                plt.title('Loss Statistics')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(process_dir, 'loss_analysis.png'), dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"✅ Loss analysis saved to {process_dir}")
            
        except Exception as e:
            print(f"⚠️  Loss analysis save failed: {e}")

    def _optimization_loop(self, noise_shape, conditioning, device, num_optimization_steps=100, 
                          learning_rate=0.05, cfg_scale=7.5, optimizer_type="Adam", debug_save_interval=100, **kwargs):
        """优化循环 - 替换原来的 scheduler.sample()"""
        
        # 初始化 latents
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
        
        print(f"[INFO] Starting SDS optimization: {num_optimization_steps} steps, lr={learning_rate}, cfg={cfg_scale}")
        
        # Debug信息初始化
        if self.debug_enabled:
            self.debug_losses = []
            print(f"🐛 Debug tracking enabled, save interval: {debug_save_interval}")
        
        # 优化循环
        for i in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # 计算 SDS loss
            loss = self._sds_loss(latents, conditioning, cfg_scale=cfg_scale)
            
            loss.backward()
            optimizer.step()
            
            # Debug保存（使用类似原版的间隔）
            self._save_debug_step(
                step=i, 
                loss=loss.item(), 
                latents=latents,
                conditioning=conditioning,
                save_interval=debug_save_interval
            )
            
            # 进度日志
            if i % max(1, num_optimization_steps // 10) == 0:
                print(f"[PROGRESS] Step {i}/{num_optimization_steps} - Loss: {loss.item():.6f}")
        
        print(f"[INFO] Optimization completed!")
        
        return latents.detach()

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, List[str]] = "",
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # === DynamiCrafter 标准参数 ===
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 1.0,  # 修正默认值
        frame_stride: int = 24,  # 修正默认值
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "tensor",
        return_dict: bool = True,
        # === Guidance 参数 ===
        num_optimization_steps: int = 100,
        learning_rate: float = 0.05,
        loss_type: str = "sds",
        weight_type: str = "t",
        cfg_scale: Optional[float] = None,
        optimizer_type: str = "Adam",
        debug_save_interval: int = 100,  # 新增：debug保存间隔
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Generate video using DynamiCrafter with guidance optimization.
        
        大部分逻辑来自 dynamicrafter_pipeline.py，只替换 scheduler.sample() 部分为 optimization loop
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
        
        # 处理 cfg_scale 参数
        if cfg_scale is None:
            cfg_scale = guidance_scale
        
        print(f"🎬 DynamiCrafter Guidance - Prompt: {prompt[0]}")
        print(f"🔧 Parameters: steps={num_optimization_steps}, lr={learning_rate}, loss={loss_type}, cfg={cfg_scale}")
        print(f"💻 Device: {device}")
        
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
        print(f"📊 Noise shape: {noise_shape}")
        
        # 准备配置信息（用于debug）
        config_info = {
            'prompt': prompt[0] if isinstance(prompt, list) else prompt,
            'negative_prompt': negative_prompt[0] if negative_prompt else "None",
            'num_optimization_steps': num_optimization_steps,
            'learning_rate': learning_rate,
            'cfg_scale': cfg_scale,
            'optimizer_type': optimizer_type,
            'loss_type': loss_type,
            'weight_type': weight_type,
            'resolution': f"{height}x{width}",
            'num_frames': num_frames,
            'frame_stride': frame_stride,
            'device': device,
            'noise_shape': str(noise_shape),
            'debug_save_interval': debug_save_interval,
            'batch_size': batch_size,
            'channels': channels,
            'latent_height': latent_height,
            'latent_width': latent_width
        }
        
        # 创建debug结构
        self._create_debug_structure(prompt[0], config_info)
        
        # 保存输入文件
        self._save_input_files(image, processed_image, prompt[0])
        
        # 保存参数配置
        self._save_parameters(config_info)
        
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
                batch_size=batch_size,
                num_frames=num_frames
            )
        
        # ===== 关键替换点：用 optimization loop 替换 scheduler.sample() =====
        samples = self._optimization_loop(
            noise_shape=noise_shape,
            conditioning=conditioning,
            device=device,
            num_optimization_steps=num_optimization_steps,
            learning_rate=learning_rate,
            cfg_scale=cfg_scale,
            optimizer_type=optimizer_type,
            debug_save_interval=debug_save_interval,
            generator=generator,
            **kwargs
        )
        
        # 解码 latents 到视频
        print(f"🎞️ Decoding latents...")
        videos = self._decode_latents(samples)
        
        print(f"✅ Video generated! Shape: {videos.shape}")
        
        # 保存最终结果和完整debug信息
        if self.debug_enabled:
            self._save_final_outputs(videos)
            self._save_process_video()
            self._save_loss_analysis()
            print(f"🐛 Complete debug results saved to: {self.debug_dir}")
        
        # 返回结果
        if return_dict:
            return {"videos": videos}
        else:
            return videos


# 测试函数
def test_sds_logic_with_complete_debug():
    """测试 SDS 逻辑 + 完整Debug功能"""
    print("🧪 Testing SDS Logic with Complete Debug Structure")
    print("=" * 60)
    
    try:
        # 启用debug模式
        pipeline = DynamiCrafterGuidancePipeline(
            resolution='256_256',
            debug_dir='./results_dynamicrafter_guidance'  # 使用类似原版的目录
        )
        
        # 创建测试图像
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        # 测试（使用较少步数但足够看到debug结构）
        result = pipeline(
            image=test_image,
            prompt="a person walking in the garden",
            num_optimization_steps=50,  # 足够生成多个debug点
            learning_rate=0.05,
            cfg_scale=7.5,
            debug_save_interval=20  # 每20步保存一次debug信息
        )
        
        videos = result["videos"] if isinstance(result, dict) else result
        print(f"✅ SDS test with complete debug completed! Result shape: {videos.shape}")
        
        if torch.isnan(videos).any():
            print("❌ NaN detected!")
            return False
        else:
            print("✅ No NaN issues")
            print(f"🐛 Complete debug files saved to: {pipeline.debug_dir}")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_sds_logic_with_complete_debug()
    else:
        print("📝 DynamiCrafter Guidance Pipeline - Complete Debug Version")
        print("\n🎯 Key Features:")
        print("  • 专注于核心 SDS loss 逻辑")
        print("  • 保持 DynamiCrafter 的所有核心逻辑")
        print("  • 只替换 scheduler.sample() 为 optimization loop")
        print("  • 完整debug保存功能（类似原版结构）")
        print("\n📁 Debug Structure:")
        print("  • inputs/: 原始图像、预处理图像、prompt")
        print("  • outputs/: 最终视频（MP4 + GIF）")
        print("  • process/: 优化过程视频、loss分析")
        print("  • debug/: 中间步骤（帧、视频）")
        print("  • params/: 配置参数（TXT + JSON）")
        print("\n📖 Usage:")
        print("  python guidance_pipeline.py test  # 测试完整debug功能")
        print("\n📊 SDS Loss Components:")
        print("  1. _sample_timestep(): 时间步采样")
        print("  2. _add_noise(): DDIM 噪声添加")
        print("  3. model.apply_model(): 噪声预测")
        print("  4. _apply_guidance_rescale(): CFG + guidance rescale")
        print("  5. DDIM reverse: 计算预测的原始样本")
        print("  6. SDS 梯度计算: w(t) * (latents - pred_original)")
        print("  7. 巧妙的损失构建: target = latents - grad")
