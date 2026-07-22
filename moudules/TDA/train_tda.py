

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import argparse
from omegaconf import OmegaConf

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from moudules.TDA.models.temporal_aligner import EEGVideoAlignmentModel


def temporal_alignment_loss_infonce(eeg_embs, target_embs, temperature=0.07):

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












def temporal_consistency_loss_distance_matrix(eeg_clip_embs, vae_clip_embs):
    """
    Compute a temporal distance-matrix consistency loss between EEG and VAE features. Both inputs may have shape (B, C, N, H, W).
    """

    assert eeg_clip_embs.shape == vae_clip_embs.shape, \
        f"Shape mismatch: {eeg_clip_embs.shape} vs {vae_clip_embs.shape}"

    B, C, N, H, W = eeg_clip_embs.shape
    D = C * H * W


    eeg_flat = eeg_clip_embs.permute(0, 2, 1, 3, 4).reshape(B, N, D)
    vae_flat = vae_clip_embs.permute(0, 2, 1, 3, 4).reshape(B, N, D)


    eeg_flat = F.normalize(eeg_flat, dim=-1)
    vae_flat = F.normalize(vae_flat, dim=-1)


    sim_eeg = torch.bmm(eeg_flat, eeg_flat.transpose(1, 2))
    sim_vae = torch.bmm(vae_flat, vae_flat.transpose(1, 2))


    dist_eeg = 1 - sim_eeg
    dist_vae = 1 - sim_vae


    loss = F.mse_loss(dist_eeg, dist_vae)

    return loss

































class EEGVideoDataset(Dataset):
    """
    Dataset loaded from real EEG arrays that also provides the matching video latent and CLIP embedding.
    """

    def __init__(self,
                 eeg_path: str,
                 vae_path: str,
                 total_time_len: int = 400,
                 in_channels: int = 62,):





        assert os.path.exists(eeg_path), f"EEG data file does not exist: {eeg_path}"


        eeg_data = np.load(eeg_path)
        if isinstance(eeg_data, np.lib.npyio.NpzFile):

            if 'eeg' in eeg_data:
                eeg_data = eeg_data['eeg']
            else:
                first_key = list(eeg_data.keys())[0]
                eeg_data = eeg_data[first_key]

        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        if eeg_tensor.ndim == 5:
            eeg_tensor = eeg_tensor.view(-1, eeg_tensor.shape[-2], eeg_tensor.shape[-1])
        assert eeg_tensor.ndim == 3, f"EEG data must be a 3D tensor shaped (N, C, T), but received  {eeg_tensor.shape}"

        self.eeg_data = eeg_tensor
        self.num_samples, self.in_channels, self.total_time_len = eeg_tensor.shape


        self.video_vae_latents_seq = torch.load(vae_path)
        assert self.video_vae_latents_seq.shape[0] == self.num_samples

        print(f"[RealEEGVideoDataset] Loaded EEG data: {self.eeg_data.shape}  from  {eeg_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        video_vae_latents = self.video_vae_latents_seq[idx]

        return eeg, video_vae_latents


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")


    train_dataset = EEGVideoDataset(


        eeg_path=cfg.dataset.eeg_path,
        vae_path=cfg.dataset.vae_path,








    )
    print(f"Total samples: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)


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

        patch_length=cfg.model.patch_length,
        patch_stride=cfg.model.patch_stride
    ).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs,eta_min= cfg.train.learning_rate*0.3)


    for epoch in range(cfg.train.num_epochs):
        model.train()
        total_alignment_loss = 0
        total_consistency_loss = 0
        total_combined_loss = 0

        for batch_idx, (eeg_data, video_vae_latents_seq) in enumerate(train_loader):
            eeg_data = eeg_data.to(device)
            video_vae_latents_seq = video_vae_latents_seq.to(device)


            optimizer.zero_grad()
            eeg_vae_emb_sequence, eeg_clip_emb_sequence = model(eeg_data)



            expected_frames = model.num_video_frames
            if video_vae_latents_seq.shape[2] != expected_frames:


                print(f"Video features have {video_vae_latents_seq.shape[2]} frames, while the model output has {expected_frames} frames; the counts do not match")
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
                eeg_vae_emb_sequence, video_vae_latents_seq
            ) * (1 - cfg.train.alignment_loss_weight)

            combined_loss = alignment_loss + consistency_loss

            combined_loss.backward()
            optimizer.step()

            total_alignment_loss += alignment_loss.item()
            total_consistency_loss += consistency_loss.item()
            total_combined_loss += combined_loss.item()

        schedular.step()
        avg_alignment_loss = total_alignment_loss / len(train_loader)
        avg_consistency_loss = total_consistency_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{cfg.train.num_epochs}], "
              f"lr: {schedular.get_last_lr()[0]:.6f}, "
              f"Total Loss: {avg_combined_loss:.4f}, "
              f"Alignment Loss (VAE): {avg_alignment_loss:.4f}, "
              f"Consistency Loss (VAE-MATRIX): {avg_consistency_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-Video Alignment Model Training")
    parser.add_argument("--config", type=str, default="/path/to/DynaMind-main/configs/tda.yaml", help="Path to the configuration file")
    args = parser.parse_args()


    cfg = OmegaConf.load(args.config)

    train(cfg)