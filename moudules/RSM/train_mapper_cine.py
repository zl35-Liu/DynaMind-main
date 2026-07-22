import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf


from moudules.RSM.models.reign_mapper_cine import UnifiedEEGModelForSeqText
from utils.training import in_batch_contrastive_loss, compute_global_mean_var



class EEGTextNPYDataset(Dataset):
    """
    Load dataset shards dynamically in batches. EEG files are stored individually, while text files contain groups of text embeddings with the documented sequence and feature dimensions.
    """
    def __init__(self, eeg_path, text_path, total_eeg_samples=5400, eeg_per_text_file=200, seq_len=50):
        self.eeg_path = eeg_path
        self.text_path = text_path
        self.total_eeg_samples = total_eeg_samples
        self.eeg_per_text_file = eeg_per_text_file


        self.eeg_indices = list(range(total_eeg_samples))
        self.length = total_eeg_samples
        self.seq_len = seq_len


        expected_text_files = total_eeg_samples // eeg_per_text_file
        if total_eeg_samples % eeg_per_text_file != 0:
             print("⚠️ Warning: total samples do not match the per-file sample count.")
        print(f"Total samples: {self.length}. Expected text files: {expected_text_files}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):


        eeg_idx = self.eeg_indices[idx]


        text_idx = eeg_idx // self.eeg_per_text_file
        sub_idx = eeg_idx % self.eeg_per_text_file




        eeg_file = os.path.join(self.eeg_path, f"{eeg_idx}.npy")
        eeg_data = np.load(eeg_file)
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        eeg_data = eeg_data[:64,]





        text_file = os.path.join(self.text_path, f"emb_simplified_{text_idx}.npy")
        text_embs = np.load(text_file)
        text_embs = text_embs[:, :self.seq_len, :]


        text_emb = text_embs[sub_idx]
        text_emb = torch.tensor(text_emb, dtype=torch.float32)

        return eeg_data, text_emb


class CosineSimLoss(nn.Module):
    """
    Sequence-level cosine-similarity loss between EEG outputs and text embeddings.
    """
    def __init__(self):
        super().__init__()

    def forward(self, eeg_emb, text_emb):




        sim = F.cosine_similarity(eeg_emb, text_emb, dim=-1)

        return 1 - sim.mean()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch,
                save_embs=False):
    model.train()
    total_loss = 0
    saving = None
    for eeg, text in tqdm(dataloader, desc="Training", leave=False):
        eeg, text = eeg.to(device), text.to(device)
        optimizer.zero_grad()


        shared, text_emb, _, _ = model(eeg)
        if save_embs:
            if saving is None:
                saving = shared.detach().cpu()
            else:
                if saving.shape[0] < 200:
                    saving = torch.cat((saving, shared.detach().cpu()), dim=0)


        if text_emb is not None:
            loss = criterion(text_emb, text)
        else:

            loss = criterion(shared, text)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if saving is not None:
        saving = saving[:200,]
        saving = saving.numpy()

        root = "/path/to/DynaMind-main/outputs/RSM/cine_embs/4/"
        os.makedirs(root, exist_ok=True)
        np.save(f"{root}eeg_embeddings{epoch}.npy", saving)
        print(f"💾 Saved EEG embeddings shape {saving.shape} to {root}eeg_embeddings.npy")
    return total_loss / len(dataloader)


def main(cfg_path="/path/to/DynaMind-main/configs/mapper_cine.yaml"):
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(cfg)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")



    dataset = EEGTextNPYDataset(
        eeg_path=cfg.data.eeg_path,
        text_path=cfg.data.text_path,
        total_eeg_samples=cfg.data.total_eeg_samples,
        eeg_per_text_file=cfg.data.eeg_per_text_file,
        seq_len=cfg.model.encoder_cfg.target_seq_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )

    model = UnifiedEEGModelForSeqText(
        cfg=cfg.model.cfg,
        encoder_cfg=cfg.model.encoder_cfg
    ).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    criterion = CosineSimLoss()



    os.makedirs(cfg.training.save_dir, exist_ok=True)


    for epoch in range(cfg.training.epochs):
        if epoch == 2 or epoch == 5 or epoch == 8:
            loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch, save_embs=True)
        else:
            loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch, save_embs=False)
        print(f"🧠 Epoch [{epoch+1}/{cfg.training.epochs}], Loss: {loss:.4f}")


        if (epoch + 1) % cfg.training.save_every == 0:
            save_path = os.path.join(cfg.training.save_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()