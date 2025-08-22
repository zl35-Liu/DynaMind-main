import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from omegaconf import OmegaConf

# 导入修改后的模型
from moudules.RSM.models.reign_mapper import UnifiedEEGModel
from utils.training import (
    in_batch_contrastive_loss, compute_global_mean_var
)


# --- 辅助函数：数据预处理 (已移至全局) ---
def preprocess_eeg_data(eeg_data_block, eeg_global_mean, eeg_global_var, device):
    """预处理EEG数据块，转换为时域张量并标准化。"""
    if eeg_data_block.ndim == 5:
        eeg_data_block = rearrange(eeg_data_block, 'a b c d e -> (a b c) d e')
    elif eeg_data_block.ndim == 4:
        eeg_data_block = rearrange(eeg_data_block, 'b c d e -> (b c) d e')

    eeg_data_block = eeg_data_block.reshape(eeg_data_block.shape[0], eeg_data_block.shape[1], -1)

    eeg_data_block = torch.from_numpy(eeg_data_block).float()
    eeg_data_block = (eeg_data_block - eeg_global_mean) / torch.sqrt(eeg_global_var)
    return eeg_data_block.to(device)


def preprocess_embedding(emb_data_block, emb_dim, device):
    """预处理嵌入数据块，转换为张量。"""
    emb_data_block = torch.from_numpy(emb_data_block).float()
    emb_data_block = torch.reshape(emb_data_block, (emb_data_block.shape[0], emb_dim))
    return emb_data_block.to(device)


# --- 统一的数据集类 ---
class UnifiedDataset(Dataset):
    def __init__(self, eeg_data, text_emb, image_emb, class_labels_dict):
        self.eeg_data = eeg_data
        self.text_emb = text_emb
        self.image_emb = image_emb
        self.labels_dict = class_labels_dict

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        labels = {task: torch.tensor(labels[idx]) for task, labels in self.labels_dict.items()}
        return (self.eeg_data[idx],
                self.text_emb[idx] if self.text_emb is not None else None,
                self.image_emb[idx] if self.image_emb is not None else None,
                labels)


# --- 统一的损失函数 ---
def unified_loss(
        eeg_emb, text_emb, image_emb, cls_logits, labels, cfg,
        text_emb_target, image_emb_target
):
    total_loss = 0
    text_cont_loss = 0
    image_cont_loss = 0
    cls_loss = 0

    if cfg.training_tasks.text_alignment.enabled:
        text_cont_loss = in_batch_contrastive_loss(text_emb, text_emb_target)
        total_loss += cfg.training_tasks.text_alignment.weight * text_cont_loss

    if cfg.training_tasks.image_alignment.enabled:
        image_cont_loss = in_batch_contrastive_loss(image_emb, image_emb_target)
        total_loss += cfg.training_tasks.image_alignment.weight * image_cont_loss

    if cfg.training_tasks.classification.enabled:
        for task_name, logits in cls_logits.items():
            if task_name in cfg.training_tasks.classification.tasks:
                task_loss = F.cross_entropy(logits, labels[task_name].long())
                cls_loss += cfg.loss_weights[task_name] * task_loss
        total_loss += cfg.training_tasks.classification.weight * cls_loss

    return total_loss, text_cont_loss, image_cont_loss, cls_loss


# --- 增强的训练流程 ---
def run_joint_training(
        cfg, fold_idx,
        train_eeg_raw, train_text_emb_raw, train_image_emb_raw, train_labels_dict_raw,
        val_eeg_raw, val_text_emb_raw, val_image_emb_raw, val_labels_dict_raw,
        eeg_global_mean, eeg_global_var
):
    print(f"\n--- 开始联合训练第 {fold_idx + 1}/{cfg.num_blocks} 折 ---")

    device = torch.device(cfg.device)

    train_eeg = preprocess_eeg_data(train_eeg_raw, eeg_global_mean, eeg_global_var, device)
    val_eeg = preprocess_eeg_data(val_eeg_raw, eeg_global_mean, eeg_global_var, device)

    train_text_emb = preprocess_embedding(train_text_emb_raw, cfg.training_tasks.text_alignment.emb_dim,
                                          device) if cfg.training_tasks.text_alignment.enabled else None
    val_text_emb = preprocess_embedding(val_text_emb_raw, cfg.training_tasks.text_alignment.emb_dim,
                                        device) if cfg.training_tasks.text_alignment.enabled else None

    train_image_emb = preprocess_embedding(train_image_emb_raw, cfg.training_tasks.image_alignment.emb_dim,
                                           device) if cfg.training_tasks.image_alignment.enabled else None
    val_image_emb = preprocess_embedding(val_image_emb_raw, cfg.training_tasks.image_alignment.emb_dim,
                                         device) if cfg.training_tasks.image_alignment.enabled else None

    train_labels_dict = {task: torch.from_numpy(labels).long() for task, labels in train_labels_dict_raw.items()}
    val_labels_dict = {task: torch.from_numpy(labels).long() for task, labels in val_labels_dict_raw.items()}

    dataset = UnifiedDataset(train_eeg.cpu(), train_text_emb.cpu() if train_text_emb is not None else None,
                             train_image_emb.cpu() if train_image_emb is not None else None, train_labels_dict)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataset = UnifiedDataset(val_eeg.cpu(), val_text_emb.cpu() if val_text_emb is not None else None,
                                 val_image_emb.cpu() if val_image_emb is not None else None, val_labels_dict)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = UnifiedEEGModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_learning_rate
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_val_loss = float('inf')
    model_save_full_dir = os.path.join(cfg.data_dir, cfg.model_save_dir)
    best_model_path = os.path.join(model_save_full_dir, f"best_model_sub{cfg.subject_id}_fold{fold_idx + 1}.pth")

    final_eeg_embs = []
    final_text_embs = []
    final_image_embs = []

    for epoch in tqdm(range(cfg.epochs), desc=f"折叠 {fold_idx + 1} 轮次"):
        model.train()
        total_loss = 0
        for eeg_batch, text_emb_target, image_emb_target, labels in dataloader:
            eeg_batch = eeg_batch.to(device)
            if text_emb_target is not None: text_emb_target = text_emb_target.to(device)
            if image_emb_target is not None: image_emb_target = image_emb_target.to(device)

            labels_dict_batch = {}
            if cfg.training_tasks.classification.enabled:
                labels_dict_batch = {task: labels[task].to(device).long() for task in
                                     cfg.training_tasks.classification.tasks.keys()}

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                eeg_emb, text_emb, image_emb, cls_logits = model(eeg_batch)
                total_loss_train, _, _, _ = unified_loss(
                    eeg_emb, text_emb, image_emb, cls_logits, labels_dict_batch, cfg,
                    text_emb_target, image_emb_target
                )

            scaler.scale(total_loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += total_loss_train.item()

        # ======= 验证阶段 =======
        if (epoch + 1) % cfg.validate_every_n_epochs == 0 or epoch == cfg.epochs - 1:
            model.eval()
            val_loss = 0
            val_correct = {}
            if cfg.training_tasks.classification.enabled:
                val_correct = {task_name: 0 for task_name in cfg.training_tasks.classification.tasks.keys()}

            with torch.no_grad():
                for eeg_val, text_val_target, image_val_target, labels_val in val_loader:
                    eeg_val = eeg_val.to(device)
                    if text_val_target is not None: text_val_target = text_val_target.to(device)
                    if image_val_target is not None: image_val_target = image_val_target.to(device)

                    labels_dict_val_batch = {}
                    if cfg.training_tasks.classification.enabled:
                        labels_dict_val_batch = {task: labels_val[task].to(device).long() for task in
                                                 cfg.training_tasks.classification.tasks.keys()}

                    eeg_emb, text_emb, image_emb, cls_logits = model(eeg_val)

                    if epoch == cfg.epochs - 1:
                        final_eeg_embs.append(eeg_emb.cpu().numpy())
                        if text_emb is not None: final_text_embs.append(text_emb.cpu().numpy())
                        if image_emb is not None: final_image_embs.append(image_emb.cpu().numpy())

                    total_loss_val, _, _, _ = unified_loss(
                        eeg_emb, text_emb, image_emb, cls_logits, labels_dict_val_batch, cfg,
                        text_val_target, image_val_target
                    )

                    val_loss += total_loss_val.item()

                    if cfg.training_tasks.classification.enabled:
                        for task_name in cfg.training_tasks.classification.tasks.keys():
                            preds = torch.argmax(cls_logits[task_name], dim=1)
                            correct = (preds == labels_dict_val_batch[task_name]).sum().item()
                            val_correct[task_name] += correct

            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Epoch {epoch + 1}: 发现最佳模型，验证损失：{best_val_loss:.4f}。已保存至 {best_model_path}")

            acc_str = []
            if cfg.training_tasks.classification.enabled:
                for task_name in cfg.training_tasks.classification.tasks.keys():
                    epoch_val_acc = val_correct[task_name] / len(val_dataset)
                    acc_str.append(f"{task_name}: {epoch_val_acc:.2%}")

            print(f"验证结果 | Total Loss: {avg_val_loss:.4f} | " + " | ".join(acc_str))

        lr_scheduler.step()

    if final_eeg_embs:
        final_eeg_embs = np.concatenate(final_eeg_embs, axis=0)
        emb_save_full_dir = os.path.join(cfg.data_dir, cfg.emb_save_dir)
        emb_save_path_eeg = os.path.join(emb_save_full_dir,
                                         f"eeg_embs/eeg_embs_fold{fold_idx + 1}_sub{cfg.subject_id}.npy")
        np.save(emb_save_path_eeg, final_eeg_embs)

        if final_text_embs:
            final_text_embs = np.concatenate(final_text_embs, axis=0)
            emb_save_path_text = os.path.join(emb_save_full_dir,
                                              f"text_embs/text_embs_fold{fold_idx + 1}_sub{cfg.subject_id}.npy")
            np.save(emb_save_path_text, final_text_embs)

        if final_image_embs:
            final_image_embs = np.concatenate(final_image_embs, axis=0)
            emb_save_path_image = os.path.join(emb_save_full_dir,
                                               f"image_embs/image_embs_fold{fold_idx + 1}_sub{cfg.subject_id}.npy")
            np.save(emb_save_path_image, final_image_embs)

        print(f"成功保存第 {fold_idx + 1} 折的最终嵌入！")

    return best_val_loss


# --- load_all_labels 函数 (已简化) ---
def load_all_labels(cfg):
    """
    从文件加载所有分类任务的标签，并根据配置文件应用必要的调整。
    """
    meta_info_path = os.path.join(cfg.data_dir, cfg.meta_info_dir)
    all_labels_map = {}

    if cfg.training_tasks.classification.enabled:
        for task_name in cfg.training_tasks.classification.tasks.keys():
            info = cfg.label_info.get(task_name)
            if info and "file" in info:
                file_path = os.path.join(meta_info_path, info["file"])
                if os.path.exists(file_path):
                    labels = np.load(file_path)

                    # 根据配置文件中的 "adjust" 字段应用标签调整
                    if "adjust" in info:
                        adjust_type = info["adjust"]
                        if adjust_type == "-1":
                            # 将标签值减 1
                            labels = labels - 1
                        elif adjust_type.startswith("threshold_"):
                            # 对标签进行阈值处理
                            try:
                                threshold_val = float(adjust_type.split('_')[1])
                                labels = np.where(labels > threshold_val, 1, 0).astype(np.int64)
                            except (IndexError, ValueError) as e:
                                print(
                                    f"Warning: Failed to parse threshold value for '{task_name}': {adjust_type}. Error: {e}")
                                # 如果解析失败，不进行调整
                                pass
                        # 其他调整类型可以在此处添加

                    # 确保标签最终是扁平化的一维数组
                    all_labels_map[task_name] = labels.flatten()
                else:
                    print(
                        f"Error: Label file for '{task_name}' not found at {file_path}. Please run preprocess_labels.py first.")
                    raise FileNotFoundError(f"Missing label file for '{task_name}'.")

    return all_labels_map


def main(cfg):
    # 创建保存目录
    emb_save_full_dir = os.path.join(cfg.data_dir, cfg.emb_save_dir)
    model_save_full_dir = os.path.join(cfg.data_dir, cfg.model_save_dir)
    os.makedirs(os.path.join(emb_save_full_dir, 'eeg_embs'), exist_ok=True)
    if cfg.training_tasks.text_alignment.enabled:
        os.makedirs(os.path.join(emb_save_full_dir, 'text_embs'), exist_ok=True)
    if cfg.training_tasks.image_alignment.enabled:
        os.makedirs(os.path.join(emb_save_full_dir, 'image_embs'), exist_ok=True)
    os.makedirs(model_save_full_dir, exist_ok=True)

    eeg_path = os.path.join(cfg.data_dir, cfg.eeg_data_path.format(subject_id=cfg.subject_id))
    text_path = os.path.join(cfg.data_dir, cfg.text_emb_path)
    image_path = os.path.join(cfg.data_dir, cfg.image_emb_path)

    all_eegdata_raw = np.load(eeg_path)
    all_text_embedding_raw = np.load(text_path)
    all_image_embedding_raw = np.load(image_path, allow_pickle=True)

    eeg_global_mean, eeg_global_var = compute_global_mean_var(all_eegdata_raw)
    eeg_global_var_tensor = torch.tensor(eeg_global_var, dtype=torch.float32)
    eeg_global_mean_tensor = torch.tensor(eeg_global_mean, dtype=torch.float32)

    all_labels_map = load_all_labels(cfg)

    overall_best_val_loss = float('inf')

    for fold_idx in range(cfg.num_blocks):
        print(f"\n==================== 运行第 {fold_idx + 1}/{cfg.num_blocks} 折 ====================")
        val_start_idx = fold_idx * cfg.samples_per_block
        val_end_idx = (fold_idx + 1) * cfg.samples_per_block

        train_eeg_raw = np.concatenate((all_eegdata_raw[:fold_idx], all_eegdata_raw[fold_idx + 1:]), axis=0)
        val_eeg_raw = all_eegdata_raw[fold_idx]

        train_text_emb_raw = np.concatenate(
            (all_text_embedding_raw[:val_start_idx], all_text_embedding_raw[val_end_idx:]), axis=0)
        val_text_emb_raw = all_text_embedding_raw[val_start_idx:val_end_idx]

        train_image_emb_raw = np.concatenate(
            (all_image_embedding_raw[:val_start_idx], all_image_embedding_raw[val_end_idx:]), axis=0)
        val_image_emb_raw = all_image_embedding_raw[val_start_idx:val_end_idx]

        train_labels_dict_raw = {}
        val_labels_dict_raw = {}
        for task_name, all_labels in all_labels_map.items():
            train_labels_dict_raw[task_name] = np.concatenate((all_labels[:val_start_idx], all_labels[val_end_idx:]),
                                                              axis=0)
            val_labels_dict_raw[task_name] = all_labels[val_start_idx:val_end_idx]

        current_fold_loss = run_joint_training(
            cfg, fold_idx,
            train_eeg_raw, train_text_emb_raw, train_image_emb_raw, train_labels_dict_raw,
            val_eeg_raw, val_text_emb_raw, val_image_emb_raw, val_labels_dict_raw,
            eeg_global_mean_tensor, eeg_global_var_tensor
        )
        if current_fold_loss < overall_best_val_loss:
            overall_best_val_loss = current_fold_loss

    print("\n" + "=" * 50)
    print("所有折叠训练结束 - 最终性能")
    print(f"所有折叠中的最低验证损失: {overall_best_val_loss:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="E:/store/DynaMind-main/configs/mapper.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    main(cfg=cfg)