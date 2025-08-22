import os
import torch
from einops import rearrange
from typing import List
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, TensorDataset
from utils.data import split_video_into_clips  # 假设你已经有了这个工具函数


# ---
# 主脚本
# ---

def main(
        pretrained_model_path: str,
        video_paths: List[str],
        output_latents_path: str,
        device: str = "cuda",
        vae_subfolder: str = "vae",
        batch_size: int = 64,  # 新增的批次大小参数
):
    """
    主函数，用于处理多个视频文件并将其编码为 VAE 潜在表示。
    为了解决内存不足问题，该脚本现在分批次处理视频片段。

    参数:
    - pretrained_model_path (str): 预训练模型根目录的路径，其中包含 'vae' 子文件夹。
    - video_paths (List[str]): 多个视频文件路径的列表。
    - output_latents_path (str): 保存潜在表示的输出文件路径。
    - device (str): 运行编码的设备 ('cuda' 或 'cpu')。
    - vae_subfolder (str): VAE模型所在的子文件夹名称。
    - batch_size (int): 每次编码的视频片段数量，用于控制显存。
    """
    if not isinstance(video_paths, list) or not video_paths:
        raise ValueError("`video_paths` must be a non-empty list of strings.")

    print("--- 步骤 1: 视频切分和帧采样 ---")
    all_video_clips = []
    for path in video_paths:
        print(f"正在处理视频: {path}")
        clips = split_video_into_clips(path)
        all_video_clips.extend(clips)

    if not all_video_clips:
        print("没有找到可用于编码的视频片段。请检查视频文件和切割逻辑。")
        return

    print(f"\n共处理得到 {len(all_video_clips)} 个视频片段。")
    print("--- 步骤 2: VAE 编码 ---")

    # 加载 VAE 模型
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder=vae_subfolder, from_tf=True)
    vae.to(device).eval()
    vae.requires_grad_(False)

    # 将所有视频片段堆叠成一个大的张量
    # 维度: (num_clips, F, C, H, W) -> (1400, 6, 3, 288, 512)
    all_clips_tensor = torch.stack(all_video_clips, dim=0)

    # 使用 DataLoader 分批处理，确保不超出显存
    dataset = TensorDataset(all_clips_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clips_batch = batch[0]
            print(f"  正在处理批次 {batch_idx + 1}/{len(dataloader)}，批次大小: {clips_batch.shape[0]}...")

            # 归一化到 [-1, 1] 范围并移动到设备
            pixel_values = (clips_batch / 127.5 - 1.0).to(device).to(torch.float32)


            # 维度重排: (batch_size * frames, channels, height, width)
            video_length = pixel_values.shape[1]
            pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

            # VAE 编码
            latents = vae.encode(pixel_values).latent_dist.sample()

            # 维度重排回 (batch_size, C_latent, frames, H_latent, W_latent)
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)

            # 缩放潜在向量
            latents = latents * 0.18215

            all_latents.append(latents)

    # 编码完成后，将所有批次的潜在表示拼接起来
    final_latents = torch.cat(all_latents, dim=0)

    print("\nVAE 编码完成。")
    print(f"潜在表示张量形状: {final_latents.shape}")

    # --- 步骤 3: 保存结果 ---
    output_dir = os.path.dirname(output_latents_path)
    if not os.path.exists(output_dir):

        os.makedirs(output_dir, exist_ok=True)

    torch.save(final_latents, output_latents_path)
    print(f"潜在表示已保存到: {output_latents_path}")


if __name__ == "__main__":
    # --- 配置参数 ---
    PRETRAINED_MODEL_PATH = "E:/store/DynaMind-main/outputs/DGVR/checkpoints/stable-diffusion-v1-4"
    VIDEO_PATHS = [
        f'E:/store/DynaMind-main/data/Video/{k}.mp4' for k in range(1, 8)
    ]
    OUTPUT_LATENTS_PATH = "E:/store/DynaMind-main/data/vae_latents/all_video_latents.pt"

    # 运行主函数
    main(
        pretrained_model_path=PRETRAINED_MODEL_PATH,
        video_paths=VIDEO_PATHS,
        output_latents_path=OUTPUT_LATENTS_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=5  # 根据你的显存大小调整这个值
    )