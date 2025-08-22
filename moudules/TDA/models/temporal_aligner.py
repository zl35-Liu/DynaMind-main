# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn.init import trunc_normal_

# Mlp 类保持不变
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# GlobalFilter 类保持不变
class GlobalFilter(nn.Module):
    def __init__(self, in_channels, seq_len, num_filters=1):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.num_filters = num_filters
        freqs = seq_len // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(num_filters, in_channels, freqs, 2, dtype=torch.float32) * 0.02
        )

    def forward(self, x):
        B, C, T = x.shape
        assert C == self.in_channels and T == self.seq_len, \
            f"GlobalFilter 输入维度不匹配: 期望({B}, {self.in_channels}, {self.seq_len}), 实际({B}, {C}, {T})"
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        output = torch.zeros(B, C, T, device=x.device, dtype=torch.float32)
        for i in range(self.num_filters):
            weight = torch.view_as_complex(self.complex_weight[i])
            filtered = x_freq * weight.unsqueeze(0)
            filtered_time = torch.fft.irfft(filtered, n=T, dim=-1, norm='ortho')
            cos_factor = torch.cos(torch.tensor((2 * i + 1) * torch.pi / (2 * self.num_filters), device=x.device))
            output += cos_factor * filtered_time
        return output


# EEGTimePatchEncoder - 修改了切分逻辑
class EEGTimePatchEncoder(nn.Module):
    def __init__(self, in_channels=62, total_time_len=400,
                 filter_num_filters=4, patch_embed_dim=256,
                 conv_out_channels=512, drop_rate=0.3,
                 patch_length=80, patch_stride=67):
        super().__init__()

        self.in_channels = in_channels
        self.total_time_len = total_time_len
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.patch_embed_dim = patch_embed_dim
        self.conv_out_channels = conv_out_channels

        # 计算滑动窗口切分后的视频帧数 (向下取整)
        self.num_video_frames = (total_time_len - patch_length) // patch_stride + 1

        # 1. 全局时域频域处理 (GlobalFilter)
        self.global_filter_block = GlobalFilter(
            in_channels=in_channels,
            seq_len=total_time_len,
            num_filters=filter_num_filters
        )

        # 2. 时序 Patch 嵌入层
        input_dim_per_patch = in_channels * patch_length
        self.patch_embedding_layers = nn.ModuleList()
        for _ in range(self.num_video_frames):
            self.patch_embedding_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim_per_patch, patch_embed_dim),
                    nn.GELU(),
                    nn.Dropout(drop_rate)
                )
            )

        # 3. Patch 后卷积层：处理拼接后的时序 Patch 嵌入
        self.patch_conv_processor = nn.Sequential(
            nn.Conv1d(patch_embed_dim, conv_out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels // 2),
            nn.GELU(),
            nn.Conv1d(conv_out_channels // 2, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_out_channels),
            nn.GELU()
        )
        self.final_projection_dim = conv_out_channels * self.num_video_frames
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, T_total = x.shape
        assert C == self.in_channels and T_total == self.total_time_len, \
            f"EEGTimePatchEncoder 输入维度不匹配: 期望({B}, {self.in_channels}, {self.total_time_len}), 实际({B}, {C}, {T_total})"

        filtered_x = self.global_filter_block(x)
        all_patch_embeddings = []

        # 2. 使用滑动窗口进行切片和嵌入
        for i in range(self.num_video_frames):
            start_idx = i * self.patch_stride
            end_idx = start_idx + self.patch_length
            current_patch_data = filtered_x[:, :, start_idx:end_idx]
            flattened_patch = current_patch_data.reshape(B, -1)
            patch_embedding = self.patch_embedding_layers[i](flattened_patch)
            all_patch_embeddings.append(patch_embedding)

        combined_patch_embeddings = torch.stack(all_patch_embeddings, dim=1)
        conv_input = combined_patch_embeddings.permute(0, 2, 1)

        processed_conv_features = self.patch_conv_processor(conv_input)
        final_flat_features = processed_conv_features.reshape(B, -1)

        return processed_conv_features, final_flat_features


# EEGVideoAlignmentModel - 构造函数添加了新的参数
class EEGVideoAlignmentModel(nn.Module):
    def __init__(self, total_time_len=400, in_channels=62,
                 filter_num_filters=4, patch_embed_dim=256,
                 conv_out_channels=512, vae_latent_c=4, vae_latent_h=8, vae_latent_w=8,
                 clip_emb_dim=768, patch_length=80, patch_stride=67):
        super().__init__()

        # 将参数传递给 EEGTimePatchEncoder
        self.eeg_encoder = EEGTimePatchEncoder(
            in_channels=in_channels,
            total_time_len=total_time_len,
            filter_num_filters=filter_num_filters,
            patch_embed_dim=patch_embed_dim,
            conv_out_channels=conv_out_channels,
            patch_length=patch_length,
            patch_stride=patch_stride
        )
        # 获取切分后的帧数
        self.num_video_frames = self.eeg_encoder.num_video_frames
        self.conv_out_channels = conv_out_channels
        self.vae_latent_c = vae_latent_c
        self.vae_latent_h = vae_latent_h
        self.vae_latent_w = vae_latent_w
        self.vae_latent_total_dim = vae_latent_c * vae_latent_h * vae_latent_w

        self.vae_latent_proj = nn.Sequential(
            nn.Linear(conv_out_channels, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.vae_latent_total_dim)
        )
        self.vae_scale = nn.Parameter(torch.tensor(1.0))
        self.clip_emb_proj = nn.Sequential(
            nn.Linear(conv_out_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, clip_emb_dim)
        )

    def forward(self, x):
        processed_conv_features, _ = self.eeg_encoder(x)
        eeg_frame_features_reshaped = processed_conv_features.permute(0, 2, 1).reshape(-1, self.conv_out_channels)
        eeg_vae_emb_flat = self.vae_latent_proj(eeg_frame_features_reshaped)
        eeg_vae_emb_sequence = eeg_vae_emb_flat.view(
            -1, self.num_video_frames,
            self.vae_latent_c, self.vae_latent_h, self.vae_latent_w
        )
        eeg_vae_emb_sequence = eeg_vae_emb_sequence.view(
            -1, self.num_video_frames,
            self.vae_latent_c, self.vae_latent_h, self.vae_latent_w
        )
        eeg_clip_emb = self.clip_emb_proj(eeg_frame_features_reshaped)
        eeg_clip_emb = F.normalize(eeg_clip_emb, dim=-1)
        eeg_clip_emb_sequence = eeg_clip_emb.view(-1, self.num_video_frames, eeg_clip_emb.shape[-1])
        return eeg_vae_emb_sequence, eeg_clip_emb_sequence