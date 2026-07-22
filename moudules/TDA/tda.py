

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch.optim as optim
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from moudules.TDA.models.temporal_aligner import EEGVideoAlignmentModel





def cfg_get(cfg, key: str, default):
    v = OmegaConf.select(cfg, key)
    return default if v is None else v


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _load_vae_latents(path: str) -> torch.Tensor:
    """
    Load VAE latents saved as a PyTorch PT or PTH file or as a NumPy NPY or NPZ file. The resulting tensor is expected to have shape (N, 4, 6, 36, 64).
    """
    assert os.path.exists(path), f"VAE latent file does not exist: {path}"

    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):

            for k in ["latents", "vae", "vae_latents", "video_vae_latents"]:
                if k in obj:
                    obj = obj[k]
                    break
        latents = obj
        if not torch.is_tensor(latents):
            raise TypeError(f"torch.load did not return a tensor or contain a latents tensor; type={type(latents)}")
        latents = latents.float().contiguous()
        return latents

    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):

            if "latents" in arr:
                arr = arr["latents"]
            elif "vae" in arr:
                arr = arr["vae"]
            else:
                arr = arr[list(arr.keys())[0]]
        latents = torch.tensor(arr, dtype=torch.float32).contiguous()
        return latents

    raise ValueError(f"Unsupported VAE latent file format: {ext}")


def resize_pred_to_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Resize a prediction with shape (B, C, Np, Hp, Wp) to the target temporal and spatial dimensions using trilinear interpolation when necessary.
    """
    assert pred.dim() == 5 and target.dim() == 5
    if pred.shape[1] != target.shape[1]:
        raise ValueError(f"Channel dimensions do not match: prediction C={pred.shape[1]} vs target C={target.shape[1]}")

    if pred.shape[2:] == target.shape[2:]:
        return pred


    pred = F.interpolate(
        pred,
        size=target.shape[2:],
        mode="trilinear",
        align_corners=False
    )
    return pred





def temporal_alignment_loss_infonce_bcn_hw(eeg_embs, target_embs, temperature=0.07):
    """
    Compute frame-level contrastive alignment for input shaped (B, C, N, H, W) by flattening each frame into one of B times N contrastive samples.
    """
    assert eeg_embs.shape == target_embs.shape, f"shape mismatch: {eeg_embs.shape} vs {target_embs.shape}"
    B, C, N, H, W = eeg_embs.shape
    D = C * H * W


    eeg_flat = eeg_embs.permute(0, 2, 1, 3, 4).reshape(B * N, D)
    tgt_flat = target_embs.permute(0, 2, 1, 3, 4).reshape(B * N, D)

    eeg_flat = F.normalize(eeg_flat, dim=-1)
    tgt_flat = F.normalize(tgt_flat, dim=-1)

    sim = torch.matmul(eeg_flat, tgt_flat.T) / temperature
    labels = torch.arange(B * N, device=eeg_embs.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def temporal_consistency_loss_distance_matrix_bcn_hw(eeg_clip_embs, vae_clip_embs):
    """
    Compute an N by N inter-frame distance matrix for each sample shaped (B, C, N, H, W) and align the matrices with mean squared error.
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

    return F.mse_loss(dist_eeg, dist_vae)





class EEGVAEAlignDataset(Dataset):
    """
    EEG: (N, C, T) from npy/npz
    VAE latents: (N, 4, 6, 36, 64) from pt/npy
    """
    def __init__(self, eeg_path: str, vae_latents_path: str):
        assert os.path.exists(eeg_path), f"EEG data file does not exist: {eeg_path}"

        eeg_data = np.load(eeg_path)
        if isinstance(eeg_data, np.lib.npyio.NpzFile):
            if "eeg" in eeg_data:
                eeg_data = eeg_data["eeg"]
            else:
                eeg_data = eeg_data[list(eeg_data.keys())[0]]

        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        if eeg_tensor.ndim == 5:

            eeg_tensor = eeg_tensor.view(-1, eeg_tensor.shape[-2], eeg_tensor.shape[-1])
        assert eeg_tensor.ndim == 3, f"EEG must have shape (N, C, T), but received  {tuple(eeg_tensor.shape)}"

        vae_latents = _load_vae_latents(vae_latents_path)
        assert vae_latents.ndim == 5, f"VAE latents must be 5D with shape (N, 4, 6, 36, 64), but received  {tuple(vae_latents.shape)}"

        if vae_latents.shape[0] != eeg_tensor.shape[0]:
            raise ValueError(f"Sample counts do not match: EEG N={eeg_tensor.shape[0]} vs VAE N={vae_latents.shape[0]}")

        self.eeg = eeg_tensor.contiguous()
        self.vae = vae_latents.contiguous()

        self.num_samples = self.eeg.shape[0]
        self.in_channels = self.eeg.shape[1]
        self.total_time_len = self.eeg.shape[2]


        self.vae_c = self.vae.shape[1]
        self.vae_frames = self.vae.shape[2]
        self.vae_h = self.vae.shape[3]
        self.vae_w = self.vae.shape[4]

        print(f"[Dataset] EEG: {tuple(self.eeg.shape)} from {eeg_path}")
        print(f"[Dataset] VAE: {tuple(self.vae.shape)} from {vae_latents_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.eeg[idx], self.vae[idx], idx





@torch.no_grad()
def run_eval_and_collect(
    model,
    loader,
    device,
    alignment_w: float,
    temperature: float,
    save_outputs: bool,
    save_max_samples: int,
    save_dtype: torch.dtype = torch.float16,
):
    model.eval()
    total_align = 0.0
    total_cons = 0.0
    total = 0.0
    n_batches = 0


    saved = {"idx": [], "pred_vae": [], "tgt_vae": []} if save_outputs else None
    saved_count = 0

    for eeg, tgt_vae, idx in loader:
        eeg = eeg.to(device, non_blocking=True)
        tgt_vae = tgt_vae.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        pred_vae, _ = model(eeg)
        pred_vae = resize_pred_to_target(pred_vae, tgt_vae)

        align_loss = temporal_alignment_loss_infonce_bcn_hw(pred_vae, tgt_vae, temperature=temperature)
        cons_loss = temporal_consistency_loss_distance_matrix_bcn_hw(pred_vae, tgt_vae)
        loss = align_loss * alignment_w + cons_loss * (1.0 - alignment_w)

        total_align += float(align_loss.item())
        total_cons += float(cons_loss.item())
        total += float(loss.item())
        n_batches += 1


        if save_outputs and saved_count < save_max_samples:
            need = save_max_samples - saved_count
            take = min(need, pred_vae.shape[0])
            saved["idx"].append(idx[:take].detach().cpu())
            saved["pred_vae"].append(pred_vae[:take].detach().cpu().to(dtype=save_dtype))
            saved["tgt_vae"].append(tgt_vae[:take].detach().cpu().to(dtype=save_dtype))
            saved_count += take

    avg = {
        "loss": total / max(n_batches, 1),
        "align_loss": total_align / max(n_batches, 1),
        "cons_loss": total_cons / max(n_batches, 1),
    }

    if save_outputs:
        saved["idx"] = torch.cat(saved["idx"], dim=0) if saved["idx"] else torch.empty(0, dtype=torch.long)
        saved["pred_vae"] = torch.cat(saved["pred_vae"], dim=0) if saved["pred_vae"] else torch.empty(0)
        saved["tgt_vae"] = torch.cat(saved["tgt_vae"], dim=0) if saved["tgt_vae"] else torch.empty(0)
        return avg, saved

    return avg, None


def save_checkpoint(path, model, optimizer, scheduler, epoch, extra: dict):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "extra": extra,
    }
    torch.save(ckpt, path)





def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")

    seed = int(cfg_get(cfg, "train.seed", 42))
    set_seed(seed)

    out_dir = ensure_dir(cfg_get(cfg, "train.output_dir", "./outputs_align"))
    ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
    pred_dir = ensure_dir(os.path.join(out_dir, "predictions"))


    dataset = EEGVAEAlignDataset(
        eeg_path=cfg.dataset.eeg_path,
        vae_latents_path=cfg.dataset.vae_path,
    )
    N_total = len(dataset)


    val_ratio = float(cfg_get(cfg, "dataset.val_ratio", 0.1))
    val_size = int(cfg_get(cfg, "dataset.val_size", max(1, int(N_total * val_ratio))))
    val_size = max(1, min(val_size, N_total - 1))
    train_size = N_total - val_size

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
    print(f"Data split: train={len(train_set)}  val={len(val_set)}")


    train_probe_size = int(cfg_get(cfg, "train.train_probe_size", min(128, len(train_set))))
    train_probe_size = max(1, min(train_probe_size, len(train_set)))

    if isinstance(train_set, Subset):
        probe_indices = train_set.indices[:train_probe_size]
        train_probe_set = Subset(train_set.dataset, probe_indices)
    else:
        train_probe_set = Subset(train_set, list(range(train_probe_size)))


    batch_size = int(cfg_get(cfg, "train.batch_size", 8))
    num_workers = int(cfg_get(cfg, "train.num_workers", 4))
    pin_memory = bool(cfg_get(cfg, "train.pin_memory", True))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    train_probe_loader = DataLoader(train_probe_set, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=pin_memory, drop_last=False)


    in_channels = dataset.in_channels
    total_time_len = dataset.total_time_len

    vae_c, vae_frames, vae_h, vae_w = dataset.vae_c, dataset.vae_frames, dataset.vae_h, dataset.vae_w

    model = EEGVideoAlignmentModel(
        total_time_len=total_time_len,
        in_channels=in_channels,
        filter_num_filters=int(cfg_get(cfg, "model.filter_num_filters", 4)),
        patch_embed_dim=int(cfg_get(cfg, "model.patch_embed_dim", 256)),
        conv_out_channels=int(cfg_get(cfg, "model.conv_out_channels", 512)),
        vae_latent_c=int(cfg_get(cfg, "projection.vae_latent_c", vae_c)),
        vae_latent_h=int(cfg_get(cfg, "projection.vae_latent_h", vae_h)),
        vae_latent_w=int(cfg_get(cfg, "projection.vae_latent_w", vae_w)),
        clip_emb_dim=int(cfg_get(cfg, "projection.clip_emb_dim", 768)),
        patch_length=int(cfg_get(cfg, "model.patch_length", 80)),
        patch_stride=int(cfg_get(cfg, "model.patch_stride", 67)),
    ).to(device)

    print(f"[Model] encoder num_video_frames = {model.num_video_frames} (target frames={vae_frames})")
    if model.num_video_frames != vae_frames:
        print("⚠️ Prediction and target frame counts differ. The prediction will be interpolated to the target (frames, H, W) shape for loss calculation.")


    lr = float(cfg_get(cfg, "train.learning_rate", 1e-4))
    num_epochs = int(cfg_get(cfg, "train.num_epochs", 5))
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * float(cfg_get(cfg, "train.eta_min_ratio", 0.3))
    )


    alignment_w = float(cfg_get(cfg, "train.alignment_loss_weight", 0.5))
    temperature = float(cfg_get(cfg, "train.temperature", 0.07))


    save_max_samples = int(cfg_get(cfg, "train.save_max_samples_per_split", 64))
    save_dtype_str = str(cfg_get(cfg, "train.save_dtype", "float16")).lower()
    save_dtype = torch.float16 if save_dtype_str == "float16" else torch.float32

    best_val = 1e18


    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_align = 0.0
        total_cons = 0.0
        n_batches = 0

        for eeg, tgt_vae, _idx in train_loader:
            eeg = eeg.to(device, non_blocking=True)
            tgt_vae = tgt_vae.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_vae, _ = model(eeg)
            pred_vae = resize_pred_to_target(pred_vae, tgt_vae)

            align_loss = temporal_alignment_loss_infonce_bcn_hw(pred_vae, tgt_vae, temperature=temperature)
            cons_loss = temporal_consistency_loss_distance_matrix_bcn_hw(pred_vae, tgt_vae)
            loss = align_loss * alignment_w + cons_loss * (1.0 - alignment_w)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_align += float(align_loss.item())
            total_cons += float(cons_loss.item())
            n_batches += 1

        scheduler.step()

        train_avg = {
            "loss": total_loss / max(n_batches, 1),
            "align_loss": total_align / max(n_batches, 1),
            "cons_loss": total_cons / max(n_batches, 1),
        }


        val_avg, val_saved = run_eval_and_collect(
            model, val_loader, device,
            alignment_w=alignment_w,
            temperature=temperature,
            save_outputs=True,
            save_max_samples=save_max_samples,
            save_dtype=save_dtype,
        )

        probe_avg, probe_saved = run_eval_and_collect(
            model, train_probe_loader, device,
            alignment_w=alignment_w,
            temperature=temperature,
            save_outputs=True,
            save_max_samples=save_max_samples,
            save_dtype=save_dtype,
        )

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch}/{num_epochs}] lr={lr_now:.6g} | "
            f"train: loss={train_avg['loss']:.4f} (align={train_avg['align_loss']:.4f}, cons={train_avg['cons_loss']:.4f}) | "
            f"probe: loss={probe_avg['loss']:.4f} | "
            f"val: loss={val_avg['loss']:.4f}"
        )


        epoch_dir = ensure_dir(os.path.join(pred_dir, f"epoch_{epoch:03d}"))
        torch.save(
            {
                "epoch": epoch,
                "split": "val",
                "metrics": val_avg,
                "idx": val_saved["idx"],
                "pred_vae": val_saved["pred_vae"],
                "tgt_vae": val_saved["tgt_vae"],
            },
            os.path.join(epoch_dir, "val_aligned_latents.pt")
        )
        torch.save(
            {
                "epoch": epoch,
                "split": "train_probe",
                "metrics": probe_avg,
                "idx": probe_saved["idx"],
                "pred_vae": probe_saved["pred_vae"],
                "tgt_vae": probe_saved["tgt_vae"],
            },
            os.path.join(epoch_dir, "train_probe_aligned_latents.pt")
        )


        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            extra={
                "train_avg": train_avg,
                "val_avg": val_avg,
                "probe_avg": probe_avg,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }
        )


        if val_avg["loss"] < best_val:
            best_val = val_avg["loss"]
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extra={
                    "best_val_loss": best_val,
                    "train_avg": train_avg,
                    "val_avg": val_avg,
                    "probe_avg": probe_avg,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                }
            )

    print(f"Training complete. Best validation loss =  {best_val:.6f}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG-Video VAE Latent Alignment Training")
    parser.add_argument("--config", type=str, default="/path/to/DynaMind-main/moudules/TDA/tda.yaml",
                        help="Path to config yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg)
