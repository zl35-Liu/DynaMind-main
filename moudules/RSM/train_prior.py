import argparse
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import yaml
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from omegaconf import OmegaConf
from collections import ChainMap


from moudules.RSM.models.diffusion_prior import BrainDiffusionPrior, PriorNetwork
from moudules.RSM.models.reign_mapper import UnifiedEEGModel


clip_seq_len = 77
clip_feature_dim = 768
EEG_ENCODER_TASK_CONFIG = {
    "video_category": 40, "face_appearance": 2, "human_appearance": 2,
    "object_count": 3, "flow": 2, "color": 7, "concept": 9,
}



class EEGTextDataset(Dataset):
    def __init__(self, raw_eeg_data_np, text_embeds_np):
        self.raw_eeg_data = torch.tensor(raw_eeg_data_np, dtype=torch.float32)
        self.text_embeds = torch.tensor(text_embeds_np, dtype=torch.float32).reshape(-1, clip_seq_len, clip_feature_dim)

    def __len__(self):
        return len(self.raw_eeg_data)

    def __getitem__(self, idx):
        return {
            "raw_eeg": self.raw_eeg_data[idx],
            "text_embed": self.text_embeds[idx]
        }


def train_diffusion_prior(config):

    train_cfg = config['train']
    model_cfg = config['model']
    pretrained_cfg = config['pretrained_encoder']
    dataset_cfg = config['dataset']
    optimizer_cfg = config['optimizer']


    guidance_cfg = config.get('guidance', {})
    guidance_types = guidance_cfg.get('types', config.get('guidance_types', []))

    accelerator = Accelerator(
        mixed_precision=train_cfg['mixed_precision'],
        gradient_accumulation_steps=train_cfg['grad_accum_steps']
    )
    device = accelerator.device

    print("Loading and freezing the pre-trained EEG encoder...")


    original_encoder_cfg_path = pretrained_cfg['original_config']


    if not os.path.exists(original_encoder_cfg_path):
        raise FileNotFoundError(f"Original encoder config not found at: {original_encoder_cfg_path}")

    with open(original_encoder_cfg_path, 'r') as f:
        original_cfg = yaml.safe_load(f)

    eeg_encoder_model = UnifiedEEGModel(OmegaConf.create(original_cfg))





    eeg_encoder_model.to(device)

    for param in eeg_encoder_model.parameters():
        param.requires_grad = False
    eeg_encoder_model.eval()
    print("Pre-trained EEG encoder loaded and frozen.")


    prior_network = PriorNetwork(

        depth=model_cfg['depth'],
        dim_head=model_cfg['dim_head'],
        heads=model_cfg['model_dim'] // model_cfg['dim_head'],
        causal=False,
        num_tokens=clip_seq_len,
        learned_query_mode=model_cfg['learned_query_mode']
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=model_cfg['model_dim'],
        condition_on_text_encodings=False,
        timesteps=model_cfg['timesteps'],
        image_embed_scale=None
    )
    model = diffusion_prior


    raw_eeg_data = np.load(dataset_cfg['raw_eeg_path'])
    text_embeds = np.load(dataset_cfg['text_embeds_path'])

    dataset = EEGTextDataset(raw_eeg_data, text_embeds)
    dataset_size = len(dataset)
    val_size = int(dataset_cfg['val_split'] * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True,
                              num_workers=dataset_cfg['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False, pin_memory=True,
                            num_workers=dataset_cfg['num_workers'])

    print(f"Dataset size: total  {dataset_size}, training  {train_size}, validation  {val_size}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_cfg['prior_lr'],
                                  weight_decay=optimizer_cfg['weight_decay'])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer_cfg['prior_lr'],
        total_steps=train_cfg['num_epochs'] * steps_per_epoch,
        pct_start=optimizer_cfg['pct_start']
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    eeg_encoder = eeg_encoder_model.eeg_encoder
    text_projection = eeg_encoder_model.text_projection
    image_projection = eeg_encoder_model.image_projection

    model.train()
    eeg_encoder.eval()
    global_step = 0
    progress_bar = tqdm(range(train_cfg['num_epochs']), desc="Training Diffusion Prior")

    for epoch in progress_bar:
        epoch_loss = 0.0
        for batch in train_loader:
            with accelerator.accumulate(model):
                eeg_features = eeg_encoder(batch["raw_eeg"])
                eeg_text_emb = text_projection(eeg_features).unsqueeze(1)
                eeg_image_emb = image_projection(eeg_features).unsqueeze(1)

                text_embed_input = eeg_text_emb if 'text_embed' in guidance_types else None
                image_embed_cond_input = eeg_image_emb if 'image_embed_cond' in guidance_types else None
                contra_embed_input = eeg_text_emb if 'contra_embed' in guidance_types else None

                loss, _ = model(
                    image_embed=batch["text_embed"],
                    text_embed=text_embed_input,
                    image_embed_cond=image_embed_cond_input,
                    contra_embed=contra_embed_input,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(accelerator.unwrap_model(model).parameters(),
                                                train_cfg['grad_norm_clip'])

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % train_cfg['log_interval'] == 0:
                    progress_bar.set_postfix(loss=loss.item())


        if epoch % train_cfg['val_interval'] == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    eeg_features = eeg_encoder(batch["raw_eeg"])
                    eeg_text_emb = text_projection(eeg_features).unsqueeze(1)
                    eeg_image_emb = image_projection(eeg_features).unsqueeze(1)

                    text_embed_input = eeg_text_emb if 'text_embed' in guidance_types else None
                    image_embed_cond_input = eeg_image_emb if 'image_embed_cond' in guidance_types else None
                    contra_embed_input = eeg_text_emb if 'contra_embed' in guidance_types else None

                    loss, _ = model(
                        image_embed=batch["text_embed"],
                        text_embed=text_embed_input,
                        image_embed_cond=image_embed_cond_input,
                        contra_embed=contra_embed_input,
                    )
                    val_loss += loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            print(f"Epoch {epoch}: Train Loss = {epoch_loss/len(train_loader):.4f}, Val Loss = {avg_val_loss:.4f}")
            model.train()


        if epoch % train_cfg['ckpt_interval'] == 0:
            checkpoint_dir = os.path.join(train_cfg['output_dir'], f"checkpoint-epoch-{epoch}")
            accelerator.save_state(output_dir=checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch}")


    final_output_dir = os.path.join(train_cfg['output_dir'], "final")
    accelerator.save_state(output_dir=final_output_dir)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Prior Training")
    parser.add_argument("--config_path", type=str, default="/path/to/DynaMind-main/configs/prior.yaml",
                        help="Path to the training configuration file (YAML or JSON).")
    args = parser.parse_args()

    config = {}
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found at: {args.config_path}")

    if args.config_path.endswith(('.yaml', '.yml')):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif args.config_path.endswith('.json'):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Unsupported config file format. Please use .yaml/.yml or .json.")

    train_diffusion_prior(config)