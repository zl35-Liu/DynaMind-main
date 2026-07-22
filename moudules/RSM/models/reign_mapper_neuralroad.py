import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange
import yaml
import os


class GatingModule(nn.Module):
    def __init__(self, in_features, target_channels):
        super().__init__()


        self.fc = nn.Linear(in_features, target_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, upstream_feature):

        weights = self.fc(upstream_feature)
        weights = self.sigmoid(weights)



        return weights.unsqueeze(-1).unsqueeze(-1)


class RegionEncoderModule_Gated(nn.Module):
    def __init__(self, current_region_channels, region_output_dim=256):
        super().__init__()


        out_h_1 = current_region_channels - 3 + 1
        out_h_2 = (out_h_1 - 3) // 2 + 1
        final_spatial_kernel = out_h_2




        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 2), padding='valid')
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()


        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding='valid')
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(final_spatial_kernel, 3),
                               stride=(1, 1), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.GELU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()


    def forward(self, x, gate_weights=None):



        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)


        if gate_weights is not None:

            x = x * gate_weights


        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.pool(x)
        x = self.flatten(x)
        return x


class RegionalEEGEncoder_Gated(nn.Module):
    def __init__(self, in_channels=62, time_len=400, region_configs=None):
        super().__init__()
        self.region_configs = region_configs or BRAIN_REGIONS_INDICES
        self.region_map = {name: i for i, (_, _, name) in enumerate(self.region_configs)}
        self.region_output_dim = 256
        self.conv1_channels = 64


        self.all_region_encoders = nn.ModuleList()
        for start_ch, end_ch, _ in self.region_configs:
            ch_i = end_ch - start_ch + 1
            self.all_region_encoders.append(RegionEncoderModule_Gated(ch_i))




        self.gating_O_to_P = GatingModule(self.region_output_dim, self.conv1_channels)


        self.gating_O_to_CM = GatingModule(self.region_output_dim, self.conv1_channels)


        self.gating_CM_to_F = GatingModule(self.region_output_dim, self.conv1_channels)

        self.fused_dim = len(self.region_configs) * self.region_output_dim


        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768)
        )
        self.apply(self._init_weights)



    def forward(self, x):

        idx_F = self.region_map['Frontal']
        idx_CM = self.region_map['Central/Motor']
        idx_P = self.region_map['Parietal']
        idx_O = self.region_map['Occipital']

        final_features = [None] * len(self.region_configs)



        O_data = x[:, 51:62, :].unsqueeze(1)
        V_O = self.all_region_encoders[idx_O](O_data)
        final_features[idx_O] = V_O




        P_data = x[:, 35:51, :].unsqueeze(1)
        M_O_to_P = self.gating_O_to_P(V_O)
        V_P = self.all_region_encoders[idx_P](P_data, gate_weights=M_O_to_P)
        final_features[idx_P] = V_P


        CM_data = x[:, 17:35, :].unsqueeze(1)
        M_O_to_CM = self.gating_O_to_CM(V_O)
        V_CM = self.all_region_encoders[idx_CM](CM_data, gate_weights=M_O_to_CM)
        final_features[idx_CM] = V_CM


        F_data = x[:, 0:17, :].unsqueeze(1)
        M_CM_to_F = self.gating_CM_to_F(V_CM)
        V_F = self.all_region_encoders[idx_F](F_data, gate_weights=M_CM_to_F)
        final_features[idx_F] = V_F


        fused_features = torch.cat(final_features, dim=1)
        final_features_768 = self.fusion_mlp(fused_features)

        return final_features_768