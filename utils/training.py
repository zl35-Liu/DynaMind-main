
import numpy as np
from typing import Union
import torch
from tqdm import tqdm



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
    Compute an in-batch contrastive loss for CLIP-style alignment between EEG embeddings and target text or image embeddings.
    """

    eeg_emb = F.normalize(eeg_emb, dim=1)
    target_emb = F.normalize(target_emb, dim=1)



    logits_per_eeg = torch.matmul(eeg_emb, target_emb.T) / temperature


    logits_per_target = logits_per_eeg.T


    labels = torch.arange(logits_per_eeg.shape[0], device=eeg_emb.device).long()


    loss_eeg = F.cross_entropy(logits_per_eeg, labels)
    loss_target = F.cross_entropy(logits_per_target, labels)


    total_loss = (loss_eeg + loss_target) / 2

    return total_loss


def compute_global_mean_var(data_block):
    """
    Compute the global mean and variance of an EEG dataset. The input usually has shape (blocks, samples per block, ...).
    """


    data_flat = data_block.flatten()
    global_mean = np.mean(data_flat)
    global_var = np.var(data_flat)
    return global_mean, global_var


def calculate_topk_accuracy(logits, labels, k=1):
    """
    Compute top-k accuracy from prediction logits and ground-truth labels.
    """
    with torch.no_grad():
        _, topk_preds = logits.topk(k, dim=1, largest=True, sorted=True)
        labels_expanded = labels.view(-1, 1).expand_as(topk_preds)
        correct = topk_preds.eq(labels_expanded).sum().item()
        accuracy = correct / labels.size(0)
    return accuracy


def plot_loss_curves(train_losses, val_losses, save_path="loss_curves.png"):
    """
    Plot the training and validation loss curves and save the figure.
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
    Plot training and validation accuracy curves for a specified task and save them in the output directory.
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