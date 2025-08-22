import argparse
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import yaml
import json
import os
from omegaconf import OmegaConf  # 确保已安装
from collections import ChainMap

# --- 导入你需要的模型和工具函数 ---
from EEG2Video.diffusion_prior.model import BrainDiffusionPrior, PriorNetwork
from EEG2Video.models.eeg_encoder_model import UnifiedEEGModel

# --- 固定配置和常量 ---
clip_seq_len = 77
clip_feature_dim = 768
EEG_ENCODER_TASK_CONFIG = {
    "video_category": 40, "face_appearance": 2, "human_appearance": 2,
    "object_count": 3, "flow": 2, "color": 7, "concept": 9,
}


# 创建数据集
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
    # 将配置字典平铺，方便访问
    flat_config = dict(ChainMap(*config.values()))

    accelerator = Accelerator(
        mixed_precision=flat_config['mixed_precision'],
        gradient_accumulation_steps=flat_config['grad_accum_steps']
    )
    device = accelerator.device

    print("Loading and freezing the pre-trained EEG encoder...")
    # 假设你的 encoder_cfg 是一个 OmegaConf 对象，或者是一个字典
    encoder_cfg = OmegaConf.create({
        "training_tasks": {
            "text_alignment": {"enabled": True, "emb_dim": clip_feature_dim},
            "image_alignment": {"enabled": True, "emb_dim": clip_feature_dim},
            "classification": {"enabled": False}
        }
    })

    eeg_encoder_model = UnifiedEEGModel(encoder_cfg)
    eeg_encoder_model.load_state_dict(torch.load(flat_config["model_path"], map_location='cpu'))
    eeg_encoder_model.to(device)

    for param in eeg_encoder_model.parameters():
        param.requires_grad = False
    eeg_encoder_model.eval()
    print("Pre-trained EEG encoder loaded and frozen.")

    prior_network = PriorNetwork(
        dim=flat_config['model_dim'],
        depth=flat_config['depth'],
        dim_head=flat_config['dim_head'],
        heads=flat_config['model_dim'] // flat_config['dim_head'],
        causal=False,
        num_tokens=clip_seq_len,
        learned_query_mode=flat_config['learned_query_mode']
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=flat_config['model_dim'],
        condition_on_text_encodings=False,
        timesteps=flat_config['timesteps'],
        image_embed_scale=None
    )
    model = diffusion_prior

    raw_eeg_data = np.load(flat_config['raw_eeg_path'])
    text_embeds = np.load(flat_config['text_embeds_path'])

    dataset = EEGTextDataset(raw_eeg_data, text_embeds)
    dataset_size = len(dataset)
    val_size = int(flat_config['val_split'] * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=flat_config['batch_size'], shuffle=True, pin_memory=True,
                              num_workers=flat_config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=flat_config['batch_size'], shuffle=False, pin_memory=True,
                            num_workers=flat_config['num_workers'])

    print(f"数据集大小: 总共 {dataset_size}, 训练 {train_size}, 验证 {val_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=flat_config['prior_lr'],
                                  weight_decay=flat_config['weight_decay'])
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=flat_config['prior_lr'],
        total_steps=flat_config['num_epochs'] * steps_per_epoch,
        pct_start=flat_config['pct_start']
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
    progress_bar = tqdm(range(flat_config['num_epochs']), desc="Training Diffusion Prior")

    guidance_types = flat_config.get('guidance_types', [])

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
                                                flat_config['grad_norm_clip'])

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % flat_config['log_interval'] == 0:
                    progress_bar.set_postfix(loss=loss.item())

        # ... (验证循环和保存模型逻辑)

    final_output_dir = os.path.join(flat_config['output_dir'], "final")
    accelerator.save_state(output_dir=final_output_dir)
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Prior Training")
    parser.add_argument("--config_path", type=str, default="./prior_config.yaml",
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