import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
import yaml
import os

# 假设你的 utils.py 已经存在
from utils import (
    in_batch_contrastive_loss, compute_global_mean_var
)

# 统一的脑区配置
BRAIN_REGIONS_INDICES = [
    (0, 16, "Frontal"),
    (17, 34, "Central/Motor"),
    (35, 50, "Parietal"),
    (51, 61, "Occipital")
]


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class RegionalEEGEncoder(nn.Module):
    def __init__(self, in_channels=62, time_len=400, region_configs=None):
        super().__init__()
        self.in_channels = in_channels
        self.time_len = time_len
        self.region_configs = region_configs or BRAIN_REGIONS_INDICES
        self.num_regions = len(self.region_configs)

        self.region_encoders = nn.ModuleList()
        self.region_output_dim = 256

        for start_ch, end_ch, _ in self.region_configs:
            current_region_channels = end_ch - start_ch + 1
            self.region_encoders.append(
                nn.Sequential(
                    nn.Conv1d(current_region_channels, 64, kernel_size=3, padding='same'),
                    nn.BatchNorm1d(64),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),

                    nn.Conv1d(64, 128, kernel_size=3, padding='same'),
                    nn.BatchNorm1d(128),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),

                    nn.Conv1d(128, 256, kernel_size=3, padding='same'),
                    nn.BatchNorm1d(256),
                    nn.GELU(),

                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten()
                )
            )

        self.fused_dim = self.num_regions * self.region_output_dim

        # 最终融合层的输出维度调整为 768
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        all_region_features = []
        for i, (start_ch, end_ch, _) in enumerate(self.region_configs):
            region_data = x[:, start_ch:end_ch + 1, :]
            region_feature = self.region_encoders[i](region_data)
            all_region_features.append(region_feature)

        fused_features = torch.cat(all_region_features, dim=1)
        final_features = self.fusion_mlp(fused_features)
        return final_features


# 适配新编码器的统一模型
class UnifiedEEGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.shared_encoder = RegionalEEGEncoder(
            in_channels=62, time_len=cfg.eeg_time_len, region_configs=BRAIN_REGIONS_INDICES
        )
        eeg_encoder_out_dim = 768

        self.text_projection = None
        self.image_projection = None
        self.classifiers = None
        self.scale = nn.Parameter(torch.tensor(1.0))

        if cfg.training_tasks.text_alignment.enabled:
            # 文本投影头，输入 768，输出 59136
            self.text_projection = nn.Sequential(
                nn.Linear(eeg_encoder_out_dim, cfg.training_tasks.text_alignment.emb_dim),
                nn.LayerNorm(cfg.training_tasks.text_alignment.emb_dim)
            )

        if cfg.training_tasks.image_alignment.enabled:
            # 图像投影头，输入 768，输出 768
            # 这里的输入维度和输出维度相同，保留投影头可以作为额外的层，也可以简化
            self.image_projection = nn.Sequential(
                nn.Linear(eeg_encoder_out_dim, cfg.training_tasks.image_alignment.emb_dim),
                nn.LayerNorm(cfg.training_tasks.image_alignment.emb_dim)
            )

        if cfg.training_tasks.classification.enabled:
            # 分类头，输入 768
            self.classifiers = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(eeg_encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                ) for task_name, num_classes in cfg.training_tasks.classification.tasks.items()
            })

    def forward(self, x):
        # 共享编码器输出 768 维特征
        shared_features = self.shared_encoder(x)

        text_emb, image_emb, cls_logits = None, None, None

        if self.text_projection:
            text_emb = self.text_projection(shared_features)
            text_emb = F.normalize(text_emb, dim=-1) * self.scale

        if self.image_projection:
            image_emb = self.image_projection(shared_features)
            image_emb = F.normalize(image_emb, dim=-1) * self.scale

        if self.classifiers:
            cls_logits = {
                task_name: classifier(shared_features)
                for task_name, classifier in self.classifiers.items()
            }

        return shared_features, text_emb, image_emb, cls_logits