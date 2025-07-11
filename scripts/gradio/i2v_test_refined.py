import os
import time
import sys
from omegaconf import OmegaConf
import torch
import numpy as np
from einops import repeat, rearrange
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, save_videos, get_latent_z
# 移除对复杂调度器的依赖，使用极简修复方案
# from dynamicrafter_scheduler import DynamiCrafterScheduler


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


def image_guided_synthesis_fixed(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., 
                                unconditional_guidance_scale=1.0, fs=None, **kwargs):
    """
    使用极简修复方案的图像引导合成函数
    """
    # 使用修复后的DDIMSampler
    ddim_sampler = get_fixed_ddim_sampler(model, ddim_steps, ddim_eta, verbose=False)
    
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    # 图像嵌入
    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    # 文本嵌入
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    
    # 图像条件
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        img_cat_cond = z[:,:,:1,:,:]
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]
    
    # 无条件引导
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts_uc = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts_uc)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(S=ddim_steps,
                                        conditioning=cond,
                                        batch_size=batch_size,
                                        shape=noise_shape[1:],
                                        verbose=False,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=uc,
                                        eta=ddim_eta,
                                        fs=fs,
                                        **kwargs)

        # 解码到像素空间
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    
    # batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


# 移除复杂的 batch_ddim_sampling_fixed_scheduler 函数
# def batch_ddim_sampling_fixed_scheduler(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,
#                                        cfg_scale=1.0, temporal_cfg_scale=None, scheduler=None, **kwargs):
#     """
#     使用修复后的 DynamiCrafter 调度器进行批量 DDIM 采样
#     """
#     # 复杂的实现被移除，使用极简修复方案


class Image2VideoFixedScheduler():
    """
    使用极简修复方案的 Image2Video 类
    """
    def __init__(self, result_dir='./tmp/', gpu_num=1, resolution='256_256') -> None:
        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1]))  # hw
        self.download_model()
        
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        ckpt_path = f'checkpoints/dynamicrafter_{resolution.split("_")[1]}_v1/model.ckpt'
        config_file = f'configs/inference_{resolution.split("_")[1]}_v1.0.yaml'
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False   
        
        model_list = []
        for gpu_id in range(gpu_num):
            model = instantiate_from_config(model_config)
            assert os.path.exists(ckpt_path), f"Error: checkpoint Not Found at {ckpt_path}!"
            model = load_model_checkpoint(model, ckpt_path)
            model.eval()
            model_list.append(model)
        
        self.model_list = model_list
        self.save_fps = 8

    def get_image(self, image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123):
        """
        使用极简修复方案生成视频
        
        Args:
            image: 输入图像 (numpy array)
            prompt: 文本提示
            steps: DDIM 步数
            cfg_scale: CFG 比例
            eta: DDIM eta 参数
            fs: 帧数
            seed: 随机种子
        """
        seed_everything(seed)
        transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        
        torch.cuda.empty_cache()
        print(f'🚀 开始生成: {prompt}')
        print(f'⏰ 时间: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))}')
        print(f'🔧 使用极简修复方案')
        start = time.time()
        
        gpu_id = 0
        if steps > 60:
            steps = 60 
            
        model = self.model_list[gpu_id]
        model = model.cuda()
        
        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [batch_size, channels, frames, h, w]

        # 文本条件
        with torch.no_grad(), torch.cuda.amp.autocast():
            # 图像条件
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
            img_tensor = (img_tensor / 255. - 0.5) * 2

            image_tensor_resized = transform(img_tensor)  # 3,h,w
            videos = image_tensor_resized.unsqueeze(0).unsqueeze(2)  # bchw -> bc1hw
            
            print(f"🎬 开始采样: {steps} 步, CFG={cfg_scale}, eta={eta}")
            
            # 使用极简修复方案进行推理
            batch_samples = image_guided_synthesis_fixed(
                model=model, 
                prompts=[prompt], 
                videos=videos, 
                noise_shape=noise_shape, 
                n_samples=1, 
                ddim_steps=steps, 
                ddim_eta=eta, 
                unconditional_guidance_scale=cfg_scale,
                fs=fs
            )
            
            # 准备文件名
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str = prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = 'empty_prompt'

        # 保存视频
        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        
        elapsed_time = time.time() - start
        print(f"✅ 生成完成!")
        print(f"💾 保存路径: {prompt_str}.mp4")
        print(f"⏱️  用时: {elapsed_time:.2f} 秒")
        print(f"🎯 平均每步: {elapsed_time/steps:.2f} 秒")
        
        model = model.cpu()
        return os.path.join(self.result_dir, f"{prompt_str}.mp4")
    
    def download_model(self):
        """下载模型权重"""
        REPO_ID = f'Doubiiu/DynamiCrafter_{self.resolution[1]}' if self.resolution[1] != 256 else 'Doubiiu/DynamiCrafter'
        filename_list = ['model.ckpt']
        model_dir = f'./checkpoints/dynamicrafter_{self.resolution[1]}_v1/'
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        for filename in filename_list:
            local_file = os.path.join(model_dir, filename)
            if not os.path.exists(local_file):
                print(f"📥 正在下载模型: {filename}")
                hf_hub_download(
                    repo_id=REPO_ID, 
                    filename=filename, 
                    local_dir=model_dir, 
                    local_dir_use_symlinks=False
                )
                print(f"✅ 下载完成: {filename}")


def test_fixed_scheduler():
    """测试极简修复方案"""
    print("🧪 测试极简修复方案")
    print("=" * 50)
    
    # 创建 I2V 实例
    i2v = Image2VideoFixedScheduler(result_dir='./results_minimal_fix/')
    
    # 测试图像路径
    test_image_path = 'prompts/art.png'
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图像不存在: {test_image_path}")
        return
    
    # 加载测试图像
    from PIL import Image
    import numpy as np
    
    img = Image.open(test_image_path).convert("RGB")
    img_array = np.array(img)
    
    # 生成视频
    test_prompt = "a man fishing in a boat at sunset, peaceful water, golden hour lighting"
    
    print(f"📸 输入图像: {test_image_path}")
    print(f"📝 提示词: {test_prompt}")
    
    try:
        video_path = i2v.get_image(
            image=img_array,
            prompt=test_prompt,
            steps=20,  # 减少步数以便快速测试
            cfg_scale=7.5,
            eta=0.0,   # 确定性采样
            fs=3,
            seed=42
        )
        print(f"🎉 测试成功! 视频保存在: {video_path}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 可以选择运行测试或单独使用
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_fixed_scheduler()
    else:
        # 简单使用示例 - 使用 256 分辨率
        i2v = Image2VideoFixedScheduler(resolution='256_256')
        
        # 使用 prompts/256/art.png 图片
        if os.path.exists('prompts/256/art.png'):
            from PIL import Image
            import numpy as np
            
            img = Image.open('prompts/256/art.png').convert("RGB")
            img_array = np.array(img)
            
            video_path = i2v.get_image(
                image=img_array,
                prompt='man fishing in a boat at sunset',
                steps=30,
                cfg_scale=7.5,
                eta=0.0,
                seed=123
            )
            print(f'✅ 完成! 视频路径: {video_path}')
        else:
            print("❌ 测试图像不存在: prompts/256/art.png")
            print("📝 用法示例:")
            print("  python i2v_test_refined.py test  # 运行测试")
            print("  或者按照代码中的示例使用 Image2VideoFixedScheduler 类")