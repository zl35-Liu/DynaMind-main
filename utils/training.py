# loss metrics optim model相关的
import numpy as np
from typing import Union
import torch
from tqdm import tqdm


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def in_batch_contrastive_loss(eeg_emb, target_emb, temperature=0.07):
    """
    计算批次内对比损失（in-batch contrastive loss），通常用于CLIP风格的对齐。

    Args:
        eeg_emb (torch.Tensor): EEG特征，形状为 (batch_size, feature_dim)。
        target_emb (torch.Tensor): 目标特征（如文本或图像），形状为 (batch_size, feature_dim)。
        temperature (float): 温度参数，用于调整损失的锐度。

    Returns:
        torch.Tensor: 计算出的对比损失。
    """
    # 归一化特征
    eeg_emb = F.normalize(eeg_emb, dim=1)
    target_emb = F.normalize(target_emb, dim=1)

    # 计算相似度矩阵（点积）
    # logits_per_eeg: (batch_size, batch_size)
    logits_per_eeg = torch.matmul(eeg_emb, target_emb.T) / temperature

    # logits_per_target: (batch_size, batch_size)
    logits_per_target = logits_per_eeg.T

    # 创建正样本标签，即对角线元素
    labels = torch.arange(logits_per_eeg.shape[0], device=eeg_emb.device).long()

    # 计算交叉熵损失
    loss_eeg = F.cross_entropy(logits_per_eeg, labels)
    loss_target = F.cross_entropy(logits_per_target, labels)

    # 总损失是两个方向损失的平均值
    total_loss = (loss_eeg + loss_target) / 2

    return total_loss


def compute_global_mean_var(data_block):
    """
    计算整个EEG数据集的全局均值和方差。

    Args:
        data_block (np.ndarray): EEG数据，形状通常为 (num_blocks, samples_per_block, ...)。

    Returns:
        tuple: 包含全局均值和全局方差的元组。
    """
    # 将所有数据块展平以计算全局统计量
    # 假设数据形状是 (B, S, C, T) 或 (B, C, T) 等，我们只关心其数值
    data_flat = data_block.flatten()
    global_mean = np.mean(data_flat)
    global_var = np.var(data_flat)
    return global_mean, global_var


def calculate_topk_accuracy(logits, labels, k=1):
    """
    计算给定logits的Top-k准确率。

    Args:
        logits (torch.Tensor): 模型的预测输出，形状为 (batch_size, num_classes)。
        labels (torch.Tensor): 真实标签，形状为 (batch_size,)。
        k (int): Top-k中的k值。

    Returns:
        float: Top-k准确率。
    """
    with torch.no_grad():
        _, topk_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        labels_expanded = labels.view(-1, 1).expand_as(topk_preds)
        correct = topk_preds.eq(labels_expanded).sum().item()
        accuracy = correct / labels.size(0)
    return accuracy


def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png"):
    """
    绘制训练和验证损失曲线。

    Args:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        save_path (str): 图像保存路径。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves saved to {save_path}")


def plot_accuracy_curves(task_name, train_accs, val_accs, save_dir="accuracy_curves"):
    """
    为特定任务绘制训练和验证准确率曲线。

    Args:
        task_name (str): 任务名称。
        train_accs (list): 训练准确率列表。
        val_accs (list): 验证准确率列表。
        save_dir (str): 图像保存目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label=f'Train Accuracy')
    plt.plot(val_accs, label=f'Validation Accuracy')
    plt.title(f'Accuracy Curves for {task_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f"{task_name}_accuracy.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy curves for {task_name} saved to {save_path}")