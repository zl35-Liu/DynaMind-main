import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
import yaml
import os


from utils import (
    in_batch_contrastive_loss, compute_global_mean_var
)


BRAIN_REGIONS_INDICES = [
    (0, 16, "Frontal"),
    (17, 34, "Central/Motor"),
    (35, 50, "Parietal"),
    (51, 61, "Occipital")
]

FRONTAL_INDICES = [i - 1 for i in range(1, 15)] + [i - 1 for i in range(16, 23)]
TEMPORAL_INDICES = [14] + [i - 1 for i in range(23, 34)] + [40]
PARIETAL_INDICES = [i - 1 for i in range(34, 41)] + [i - 1 for i in range(42, 51)]
OCCIPITAL_INDICES = [i - 1 for i in range(51, 63)]






















class RegionalEEGEncoder_ST_Pooling(nn.Module):
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







            out_h_1 = current_region_channels - 3 + 1




            out_h_2 = (out_h_1 - 3) // 2 + 1



            final_spatial_kernel = out_h_2


            self.region_encoders.append(
                nn.Sequential(


                    nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2), padding='valid'),
                    nn.BatchNorm2d(64),
                    nn.GELU(),



                    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding='valid'),
                    nn.BatchNorm2d(128),
                    nn.GELU(),



                    nn.Conv2d(128, 256, kernel_size=(final_spatial_kernel, 3),
                            stride=(1, 1), padding=(0, 1)),
                    nn.BatchNorm2d(256),
                    nn.GELU(),


                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
            )

        self.fused_dim = self.num_regions * self.region_output_dim


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
        elif isinstance(m, (nn.Conv1d, nn.BatchNorm1d, nn.Conv2d, nn.BatchNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        all_region_features = []
        for i, (start_ch, end_ch, _) in enumerate(self.region_configs):
            region_data = x[:, start_ch:end_ch + 1, :]



            region_data = region_data.unsqueeze(1)

            region_feature = self.region_encoders[i](region_data)
            all_region_features.append(region_feature)

        fused_features = torch.cat(all_region_features, dim=1)
        final_features = self.fusion_mlp(fused_features)
        return final_features

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

            self.text_projection = nn.Sequential(
                nn.Linear(eeg_encoder_out_dim, 512),
                nn.Linear(512,512),
                nn.Linear(512, cfg.training_tasks.text_alignment.emb_dim),
                nn.LayerNorm(cfg.training_tasks.text_alignment.emb_dim)
            )

        if cfg.training_tasks.image_alignment.enabled:


            self.image_projection = nn.Sequential(
                nn.Linear(eeg_encoder_out_dim, cfg.training_tasks.image_alignment.emb_dim),
                nn.LayerNorm(cfg.training_tasks.image_alignment.emb_dim)
            )

        if cfg.training_tasks.classification.enabled:

            self.classifiers = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(eeg_encoder_out_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                ) for task_name, num_classes in cfg.training_tasks.classification.tasks.items()
            })

    def forward(self, x):

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