# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
from omegaconf import OmegaConf

# 导入你拆分出的模型脚本
from moudules.TDA.models.temporal_aligner import EEGVideoAlignmentModel

# --- 损失函数和数据加载器保持不变 ---
def temporal_alignment_loss_infonce(eeg_embs, target_embs, temperature=0.07):
    # ... (与之前相同)
    B, N, C, H, W = eeg_embs.shape
    D_total = C * H * W
    eeg_flat = eeg_embs.view(B * N, D_total)
    target_flat = target_embs.view(B * N, D_total)
    eeg_flat_norm = F.normalize(eeg_flat, dim=-1)
    target_flat_norm = F.normalize(target_flat, dim=-1)
    similarity_matrix = torch.matmul(eeg_flat_norm, target_flat_norm.T) / temperature
    labels = torch.arange(B * N, device=eeg_embs.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def temporal_consistency_loss_distance_matrix(eeg_clip_embs, video_clip_embs):
    # ... (与之前相同)
    B, N, D = eeg_clip_embs.shape
    sim_eeg = torch.bmm(eeg_clip_embs, eeg_clip_embs.transpose(1, 2))
    dist_eeg = 1 - sim_eeg
    sim_video = torch.bmm(video_clip_embs, video_clip_embs.transpose(1, 2))
    dist_video = 1 - sim_video
    loss = F.mse_loss(dist_eeg, dist_video)
    return loss


class SimulatedEEGVideoDataset(Dataset):
    # ... (与之前相同)
    def __init__(self, groups, classes, samples_per_class,
                 in_channels, total_time_len, num_video_frames,
                 vae_latent_c, vae_latent_h, vae_latent_w, clip_emb_dim):
        self.num_samples = groups * classes * samples_per_class
        self.in_channels = in_channels
        self.total_time_len = total_time_len
        self.num_video_frames = num_video_frames
        self.vae_latent_c = vae_latent_c
        self.vae_latent_h = vae_latent_h
        self.vae_latent_w = vae_latent_w
        self.clip_emb_dim = clip_emb_dim
        simulated_5d_eeg = torch.randn(groups, classes, samples_per_class, in_channels, total_time_len)
        self.eeg_data = simulated_5d_eeg.view(self.num_samples, in_channels, total_time_len)
        self.video_vae_latents_seq = torch.randn(
            self.num_samples, num_video_frames, vae_latent_c, vae_latent_h, vae_latent_w
        )
        self.video_clip_embs_seq = F.normalize(
            torch.randn(self.num_samples, num_video_frames, clip_emb_dim), dim=-1
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        video_vae_latents = self.video_vae_latents_seq[idx]
        video_clip_embs = self.video_clip_embs_seq[idx]
        return eeg, video_vae_latents, video_clip_embs


# --- 训练函数 - 调整了模型参数的传递 ---
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"开始训练，设备: {device}")

    # 1. 数据集和数据加载器
    train_dataset = SimulatedEEGVideoDataset(
        groups=cfg.dataset.groups,
        classes=cfg.dataset.classes,
        samples_per_class=cfg.dataset.samples_per_class,
        in_channels=cfg.dataset.in_channels,
        total_time_len=cfg.dataset.total_time_len,
        num_video_frames=cfg.dataset.num_video_frames,  # 这里的参数仍然用于模拟数据生成
        vae_latent_c=cfg.projection.vae_latent_c,
        vae_latent_h=cfg.projection.vae_latent_h,
        vae_latent_w=cfg.projection.vae_latent_w,
        clip_emb_dim=cfg.projection.clip_emb_dim
    )
    print(f"总样本数: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # 2. 模型实例化
    model = EEGVideoAlignmentModel(
        total_time_len=cfg.dataset.total_time_len,
        in_channels=cfg.dataset.in_channels,
        filter_num_filters=cfg.model.filter_num_filters,
        patch_embed_dim=cfg.model.patch_embed_dim,
        conv_out_channels=cfg.model.conv_out_channels,
        vae_latent_c=cfg.projection.vae_latent_c,
        vae_latent_h=cfg.projection.vae_latent_h,
        vae_latent_w=cfg.projection.vae_latent_w,
        clip_emb_dim=cfg.projection.clip_emb_dim,
        # 传入新的滑动窗口参数
        patch_length=cfg.model.patch_length,
        patch_stride=cfg.model.patch_stride
    ).to(device)

    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    # 4. 训练循环
    for epoch in range(cfg.train.num_epochs):
        model.train()
        total_alignment_loss = 0
        total_consistency_loss = 0
        total_combined_loss = 0

        for batch_idx, (eeg_data, video_vae_latents_seq, video_clip_embs_seq) in enumerate(train_loader):
            eeg_data = eeg_data.to(device)
            video_vae_latents_seq = video_vae_latents_seq.to(device)
            video_clip_embs_seq = video_clip_embs_seq.to(device)

            optimizer.zero_grad()
            eeg_vae_emb_sequence, eeg_clip_emb_sequence = model(eeg_data)

            # 确保模拟的 video_vae_latents_seq 和 video_clip_embs_seq 的帧数与模型输出匹配
            # 这是一个关键步骤，因为模型现在切分的帧数是动态计算的
            expected_frames = model.num_video_frames
            if video_vae_latents_seq.shape[1] != expected_frames:
                # 在实际数据中，你需要确保你的视频特征与模型输出的帧数一致
                # 这里我们重新生成模拟数据以匹配
                B = eeg_data.shape[0]
                video_vae_latents_seq = torch.randn(
                    B, expected_frames, cfg.projection.vae_latent_c, cfg.projection.vae_latent_h,
                    cfg.projection.vae_latent_w
                ).to(device)
                video_clip_embs_seq = F.normalize(
                    torch.randn(B, expected_frames, cfg.projection.clip_emb_dim), dim=-1
                ).to(device)

            alignment_loss = temporal_alignment_loss_infonce(
                eeg_vae_emb_sequence, video_vae_latents_seq
            ) * cfg.train.alignment_loss_weight

            consistency_loss = temporal_consistency_loss_distance_matrix(
                eeg_clip_emb_sequence, video_clip_embs_seq
            ) * cfg.train.consistency_loss_weight

            combined_loss = alignment_loss + consistency_loss

            combined_loss.backward()
            optimizer.step()

            total_alignment_loss += alignment_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_combined_loss += combined_loss.item()

        avg_alignment_loss = total_alignment_loss / len(train_loader)
        avg_consistency_loss = total_consistency_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{cfg.train.num_epochs}], "
              f"Total Loss: {avg_combined_loss:.4f}, "
              f"Alignment Loss (VAE): {avg_alignment_loss:.4f}, "
              f"Consistency Loss (CLIP-Dist): {avg_consistency_loss:.4f}")

    print("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-Video Alignment Model Training")
    parser.add_argument("--config", type=str, default="E:/store/DynaMind-main/configs/tda.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    # 使用 OmegaConf 加载配置文件
    cfg = OmegaConf.load(args.config)

    train(cfg)