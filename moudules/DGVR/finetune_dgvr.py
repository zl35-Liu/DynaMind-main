# 导入必要的库和模块

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 在导入 accelerate 之前设置

import argparse  # 解析命令行参数
import datetime  # 处理日期和时间
import logging  # 日志记录
import inspect  # 获取对象信息
import math  # 数学运算
import os  # 操作系统相关功能
from typing import Dict, Optional, Tuple  # 类型提示
from omegaconf import OmegaConf  # 处理配置文件
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
print(f"PyTorch 版本: {torch.__version__}")          # 应输出 2.0.1+cu118
print(f"CUDA 可用: {torch.cuda.is_available()}")    # 应输出 True
print(f"CUDA 版本: {torch.version.cuda}")           # 应输出 11.8
import torch.nn.functional as F  # PyTorch神经网络函数
import torch.utils.checkpoint  # 梯度检查点

import diffusers  # 扩散模型库
import transformers  # 自然语言处理模型库
from accelerate import Accelerator  # 分布式训练加速库
from accelerate.logging import get_logger  # 获取日志记录器
from accelerate.utils import set_seed  # 设置随机种子
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler  # 扩散模型组件
from diffusers.optimization import get_scheduler  # 学习率调度器
from diffusers.utils import check_min_version  # 检查库版本
from diffusers.utils.import_utils import is_xformers_available  # 检查xformers可用性
from tqdm import tqdm  # 进度条显示
from transformers import CLIPTextModel, CLIPTokenizer  # CLIP模型

from moudules.DGVR.models.unet import UNet3DConditionModel  # 自定义3D UNet模型
from moudules.DGVR.data.dataset import TuneAVideoDataset, TuneMultiVideoDataset,TuneMultiVideoDataset1,TuneMultiVideoDataset2  # 自定义数据集类

from moudules.DGVR.models.pipeline_tuneavideo import TuneAVideoPipeline  # 自定义视频生成管道
from utils import *  # 保存视频网格和DDIM反转
from einops import rearrange  # 张量操作

# import xformers
# from xformers.ops import memory_efficient_attention
import torch
import torch.optim as optim
from PIL import Image
import decord
import  matplotlib.pyplot as plt
import statistics

def compute_global_mean_var(data):
    flattened_data = data.flatten()  # 展平为1D数组
    global_mean = np.mean(flattened_data)
    global_var = np.var(flattened_data)
    return global_mean, global_var

def save_video_frames(video_tensor, output_path):
    # 输入形状: (num_frames, C, H, W)
    frames = []
    for frame in video_tensor:
        frame = frame.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    # 保存为GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=333,  # 每帧时长（ms）
        loop=0
    )



# 设置环境变量以优化CUDA内存分配
os.environ["PYTORCH_CUDA_ALLOC_conf"] = "max_split_size_mb:24"

# 检查diffusers最低版本
check_min_version("0.10.0.dev0")

# 获取日志记录器
logger = get_logger(__name__, log_level="INFO")


def main(
        # 主函数参数定义（配置相关参数）
        pretrained_model_path: str,  # 预训练模型路径
        output_dir: str,  # 输出目录
        train_data: Dict,  # 训练数据配置
        validation_data: Dict,  # 验证数据配置
        validation_steps: int = 100,  # 验证频率
        trainable_modules: Tuple[str] = ("attn1.to_q", "attn2.to_q", "attn_temp"),  # 可训练模块
        train_batch_size: int = 2,  # 训练批次大小
        max_train_steps: int = 1200000,  # 最大训练步数（已注释）
        learning_rate: float = 3e-5,  # 学习率
        scale_lr: bool = False,  # 是否按批次等缩放学习率
        lr_scheduler: str = "constant",  # 学习率调度器类型
        lr_warmup_steps: int = 0,  # 学习率预热步数
        adam_beta1: float = 0.9,  # Adam优化器参数
        adam_beta2: float = 0.999,  # Adam优化器参数
        adam_weight_decay: float = 1e-2,  # 权重衰减
        adam_epsilon: float = 1e-08,  # Adam epsilon
        max_grad_norm: float = 5.0,  # 梯度裁剪阈值  初始1
        gradient_accumulation_steps: int = 2,  # 梯度累积步数
        gradient_checkpointing: bool = True,  # 是否启用梯度检查点
        checkpointing_steps: int = 500,  # 检查点保存步数
        resume_from_checkpoint: Optional[str] = None,  # 从检查点恢复训练
        mixed_precision: Optional[str] = "fp16",  # 混合精度训练
        use_8bit_adam: bool = False,  # 是否使用8位Adam优化器
        enable_xformers_memory_efficient_attention: bool = True,  # 启用xformers高效注意力
        seed: Optional[int] = None,  # 随机种子
):
    # 获取当前函数的参数并保存到配置
    *_, config = inspect.getargvalues(inspect.currentframe())

    # 初始化Accelerator（处理分布式训练和混合精度）
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # 配置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # 主进程设置日志级别
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if seed is not None:
        set_seed(seed)

    # 创建输出目录（主进程处理）
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))  # 保存配置

    # 加载模型组件
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")  # 噪声调度
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")  # 文本分词器
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")  # 文本编码器
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", from_tf=True)  # 变分自编码器
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")  # 3D UNet模型

    # 冻结VAE和文本编码器的参数
    vae.requires_grad_(False)             # 冻结VAE参数
    text_encoder.requires_grad_(False)    #

    # 仅解冻UNet中指定的可训练模块
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    #启用xformers高效注意力（如果可用）
    # if enable_xformers_memory_efficient_attention:
    #     if is_xformers_available():
    #         unet.enable_xformers_memory_efficient_attention()
    #     else:
    #         raise ValueError("xformers is not available. Make sure it is installed correctly")

    # 启用梯度检查点节省内存
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # 根据配置调整学习率
    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # 初始化优化器（选择8位Adam或普通Adam）
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Please install bitsandbytes to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    video_list=[]
    for k in range(1,8):
        path = f'E:/store/DynaMind-main/data/Video/{k}.mp4'
        # 加载 切分 视频
        vr = decord.VideoReader(path, width=512, height=288)

        # fps = vr.get_avg_fps()  # 获取视频帧率
        fps = 24  # 视频24帧
        total_frames = len(vr)  # 获取视频总帧数
        # print(total_frames)
        waste_time = 3
        block_time = 13
        # sample_index = []  # 存储采样帧的索引 遍历所有块逐步添加 最后导入视频
        for i in range(40):
            # 计算每个片段的起始帧和结束帧
            start = int(waste_time * fps + block_time * fps * i)  # 当前块开始的帧索引

            for j in range(5):

                clip_frame_length = int(2 * fps)  # 每个2s的帧数
                start_frame = start + clip_frame_length * j + 1  # 当前2s开始的帧索引
                end_frame = start_frame + clip_frame_length  # 当前片段的结束帧
                if end_frame > 12480:
                    end_frame = 12480
                # if(i==39):
                # print("第",i,"块的第",j,"个片段的起始帧index ",start_frame,"结束帧index ",end_frame)

                # 确保片段不超过视频总长度
                # if end_frame > total_frames + 1:
                #     raise ValueError(f"Clip {index} exceeds video length.")

                # 获取每个片段的帧
                clip = vr.get_batch(range(start_frame, end_frame))
                # print("2s片段有多少帧",len(clip))  #检查正确 2s 48帧
                clip = rearrange(clip, "f h w c -> f c h w")  # 转换为 (frames, channels, height, width)

                # 采样选定帧
                # video = clip[::self.sample_frame_rate][:self.n_sample_frames]  # 每8帧采样一次，总共采样6帧
                video = clip[::8]  # 每8帧采样一次，总共采样6帧
                video_list.append(video)

    # 加载和准备训练数据集
    # train_dataset = TuneMultiVideoDataset(**train_data)  # 多视频数据集实例
    # train_dataset1 = TuneAVideoDataset(**train_data)  # 猜测使用已有的但视频数据导入
    # train_dataset2 = TuneMultiVideoDataset1(**train_data)
    train_dataset = TuneMultiVideoDataset2(**train_data)  # 多视频数据集实例

    # 手动设置视频路径和文本提示（示例代码中可能应通过配置文件处理）
    video_path2 = '../SEED-DV/Video'
    video_path = ('../SEED-DV/Video/1.mp4')
    # video_ind = [i for i in range(1, 51)]
    # video_ind = [i for i in range(1, 8)]                                # 报错说应该传单个路径 故修改
    # video_files = [f"{video_path}/{ind}.mp4" for ind in video_ind]
    # train_dataset.video_path = video_path                        #  将视频路径列表赋给 train_dataset
    # train_dataset1.video_path = video_path                        #  将视频路径列表赋给 train_dataset
    # train_dataset2.video_path = video_path2
    train_dataset.video=video_list

    video_text2 = 'E:/store/DynaMind-main/data/Video/BLIP-caption/all.txt'
    # video_text2 = ('../SEED-DV/Video/BLIP-caption/1+2.txt')

    with open(video_text2, 'r') as f:
        text_prompts2 = [line.strip() for line in f]
        text_prompts2 = text_prompts2[:400]
    # train_dataset.prompt = text_prompts                               #  将文本路径列表赋给 train_dataset
    # train_dataset1.prompt = text_prompts                               #  将文本路径列表赋给 train_dataset
    # train_dataset2.prompt = text_prompts2
    train_dataset.prompt = text_prompts2
    print("data ok")

    '''
    自己的multi导入 传进去是一个包含若干example（字典：2s视频文本pair）的列表
    应设计循环，每次循环将一个example送入train_dataset
    '''

    # 对文本提示进行分词编码
    train_dataset.prompt_ids = tokenizer(
        list(train_dataset.prompt),
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    # train_dataset1.prompt_ids = tokenizer(
    #     list(train_dataset.prompt),
    #     max_length=tokenizer.model_max_length,
    #     padding="max_length",
    #     truncation=True,
    #     return_tensors="pt"
    # ).input_ids

    # print("应该是40或者200，len（dataset）： ",len(train_dataset))
    # for batch in train_dataset:
    #     print("type(batch): ",type(batch))  # 打印数据类型
    #     # print("batch.shape: ",batch.shape)  # 打印数据形状（例如，Tensor 的形状）
    #     print("type(batch[0]: ",type(batch[0]))
    #     print("multi_pixel: ",batch[0]['pixel_values'].shape)  # 打印数据内容（例如，Tensor 或字典等）
    #     print("multi_prompt: ",batch[0]['prompt_ids'].shape)  # 打印数据内容（例如，Tensor 或字典等）
    #
    #     break  # 只查看第一个批次
    # print("1")
    # print("type(dataset1): ", type(train_dataset1))
    # for batch in train_dataset1:
    #     print("type(batch): ", type(batch))  # 打印数据类型
    #     # print("batch.shape: ", batch.shape)  # 打印数据形状（例如，Tensor 的形状）
    #     # print("type(batch[0]: ", type(batch[0]))
    #     # print(batch)  # 打印数据内容（例如，Tensor 或字典等）
    #     print("data1_pixel: ",batch['pixel_values'].shape)  # 打印数据内容（例如，Tensor 或字典等）
    #     print("data1_prompt: ",batch['prompt_ids'].shape)  # 打印数据内容（例如，Tensor 或字典等）
    #     break  # 只查看第一个批次


    # 创建训练数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False  # 打乱数据顺序
    )

    # 准备验证管道和DDIM调度器
    validation_pipeline = TuneAVideoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()  # 启用VAE切片节省内存

    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)  # 设置DDIM反转步数

    # 计算训练周期数（示例中固定为6000）
    num_train_epochs = 10

    # 初始化学习率调度器
    # lr_scheduler = get_scheduler(
    #     lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
    #     num_training_steps=num_train_epochs * len(train_dataloader) * gradient_accumulation_steps,
    # )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_train_epochs, eta_min=5e-6
    )

    # 使用Accelerator准备模型、优化器、数据加载器等
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # 处理混合精度下的数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将文本编码器和VAE移动到对应设备并转换数据类型
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 初始化训练跟踪器（主进程）
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # 训练循环准备
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size per device = {train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

    global_step = 0
    first_epoch = 1

    torch.cuda.empty_cache()

    latent_path="/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents"  #vae视频变量存储/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents
    latents_list = []  # 用于存储所有视频的 latent
    noise_latents_list = []
    loss_list=[]
    # 开始训练循环
    for epoch in tqdm(range(first_epoch, num_train_epochs + 1)):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):   #total对应multi导入的list 其中再包含若干个example

            # print(f"Processing step {step}")
            # print("total[0] : ",total[i])
            print("pixelvalues",batch['pixel_values'].shape)

            print("promptids",batch['prompt_ids'].shape)

            with accelerator.accumulate(unet):  # 梯度累积上下文
                # 将视频像素值转换为潜在表示
                pixel_values = batch["pixel_values"].to(weight_dtype)
                print(pixel_values.shape)
                video = pixel_values/2 + 0.5
                video = video.squeeze(0)
                print("video sahpe ",video.shape)
                prompts = batch["prompt_ids"]

                input_path = f"{output_dir}/input_samples"
                if not os.path.exists(input_path):
                    os.makedirs(input_path, exist_ok=True)
                if epoch==1 and step>200:
                    save_video_frames(video, f"{input_path}/{text_prompts2[step]}.gif") # 检查输入视频质量是否正确
                # print("pixel_values.shape: ",pixel_values.shape)
                video_length = pixel_values.shape[1]
                # 重构张量维度：(batch, frames, channels, height, width) -> (batch*frames, channels, height, width)
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                print("pixel 是否归一 ",pixel_values.min(), pixel_values.max())

                # 这里latent应该对应了z0
                latents = vae.encode(pixel_values).latent_dist.sample()  # VAE编码
                # print("latents.shape: ",latents.shape)
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215  # 缩放潜在向量
                # print(f"检查Latent mean 应0: {latents.mean().item()}, std 约1: {latents.std().item()}")
                if epoch == num_train_epochs:
                    latents_list.append(latents)

                # print("latents.shape_rereange: ",latents.shape)
                # # 将每个视频的 latent 加入到列表
                # if epoch == num_train_epochs:
                #     latents_list.append(latents)

                # 生成噪声并采样时间步
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                # timesteps = torch.full((bsz,), 1, device=latents.device, dtype=torch.long)    # 固定时间步为100 测试
                # timesteps = torch.full((1,), 999, device="cuda", dtype=torch.long)

                # timesteps = torch.linspace(0,999,100).long().to(latents.device) # 另一种写法
                # print(timesteps)

                 # 前向扩散过程：添加噪声到潜在向量
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 考虑 保存noisy latents用于训练
                if epoch == num_train_epochs:
                    noise_latents_list.append(noisy_latents)

                # 获取文本编码
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]   #
                print(encoder_hidden_states.shape)
                out = encoder_hidden_states
                print(f"文本输入范围: [{out.min():.2f}, {out.max():.2f}]")  # 应≈[-3,3]
                print(f"文本嵌入范数: {out.norm(dim=-1).mean():.2f}")  # 应≈1.0
                out = out.cpu().detach().numpy()
                text_mean, text_var = compute_global_mean_var(out)
                print("文本embedding总体均值:", text_mean)
                print("文本embedding总体方差:", text_var)


                # 根据调度器类型确定目标
                if noise_scheduler.prediction_type == "epsilon":    # 默认代码选择的是epsilon
                    target = noise
                    # print("target noise")
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    # print("target get_velocity")
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # 预测噪声并计算损失
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # print("pred shape ",model_pred.shape)  # 1, 4, 6, 36, 64
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                print(f"epoch {epoch} Processing step {step} loss {loss.item()} lr {lr_scheduler.get_last_lr()[0]}")


                # # 收集各进程的损失并平均
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps
                loss_list.append(train_loss)

                # 反向传播和优化
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)  # 梯度裁剪
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()


            # 同步梯度后更新全局步数
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                # print("bushu")

            # 记录当前步骤的损失和学习率
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        #
        lr_scheduler.step()
        # print(epoch, "epoch_loss", loss.detach().item(), "lr", lr_scheduler.get_last_lr()[0])
        # 验证和保存模型（每1200个epoch执行）
        if epoch % 1 == 0:     #先改为50
            if accelerator.is_main_process:
                samples = []
                generator = torch.Generator(device=latents.device).manual_seed(seed)

                # DDIM潜在向量反转（如果启用）
                ddim_inv_latent = None
                if validation_data.use_inv_latent:
                    inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{epoch}.pt")
                    ddim_inv_latent = ddim_inversion(
                        validation_pipeline, ddim_inv_scheduler,
                        video_latent=latents,
                        num_inv_steps=validation_data.num_inv_steps,
                        prompt=""
                    )[-1].to(weight_dtype)
                    # torch.save(ddim_inv_latent, inv_latents_path)

                # 生成验证样本
                for idx, prompt in enumerate(validation_data.prompts):
                    sample = validation_pipeline(
                        prompt,
                        generator=generator,
                        latents=None,
                        **validation_data
                    ).videos
                    save_videos_grid(sample, f"{output_dir}/samples/sample-{epoch}/{prompt}.gif")
                    samples.append(sample)

                # 保存所有样本的网格视频
                samples = torch.concat(samples)
                save_path = f"{output_dir}/samples/sample-{epoch}.gif"
                save_videos_grid(samples, save_path)
                logger.info(f"Saved samples to {save_path}")

            # 所有进程等待主进程完成验证
            accelerator.wait_for_everyone()

            # 主进程保存模型
            if accelerator.is_main_process:
                unet = accelerator.unwrap_model(unet)
                pipeline = TuneAVideoPipeline.from_pretrained(
                    pretrained_model_path,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                )
                pipeline.save_pretrained(output_dir)

    #训练后保存latents
    # # 将所有视频的 latent 堆叠成一个张量（增加一个维度来表示不同视频）
    # if len(latents_list)==0:
    #     print("latents 没有")
    # if not os.path.exists(os.path.dirname(latent_path)):
    #     os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    #     print(f"文件夹 {os.path.dirname(latent_path)} 已创建。")
    # else:
    #     print(f"文件夹 {os.path.dirname(latent_path)} 已存在。")
    # all_latents = torch.stack(latents_list, dim=0)  # 形状变为 [num_videos, latent_dim]
    # all_noise_latents = torch.stack(noise_latents_list, dim=0)  # 形状变为 [num_videos, latent_dim]
    # print("latent shape预计是（1200（200*6），l）",all_latents.shape)
    # torch.set_printoptions(profile="full")
    # # print(all_latents)
    # # torch.save(all_latents, latent_path)
    # torch.save(all_latents, '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor9.pt')
    # torch.save(all_noise_latents, '/home/bcilab/Ljx/EEG2Video-main/EEG2Video/vae_latents/test_tensor9n.pt')
    '''
    tensor - 原始
    tensor1 - 加噪的保存
    tensor2 - 加噪的保存6epoch
    修改了数据集导入 帧对上了
    tensor3 - 不noise try8 
    tensor3n - noise
    tensor4 - 不noise try9  限定时间步100 对上inference
    tensor5 - try10 400时间步
    tensor6 - try11 800时间步  
    tensor7 - try12 50时间步 不微调   loss 0.4-0.5       100步loss 0.3-0.4
    tensor8 - try13 999时间步 不微调  
    tensor9 - try14 1时间步 
    '''
    print("torch_save_ok")


    # 训练结束保存最终模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = TuneAVideoPipeline.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    # 绘制损失曲线
    meandata = []

    for i in range(num_train_epochs):
        avg = statistics.mean(loss_list[i * (len(train_dataloader) / train_batch_size):(i + 1) * (
                    len(train_dataloader) / train_batch_size)])
        meandata.append(avg)
    plt.plot(range(meandata), meandata, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(f"{output_dir}/train_loss.gif")
    plt.show()

    accelerator.end_training()


# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="E:\store\DynaMind-main\configs\dgvr.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))


"""

"""