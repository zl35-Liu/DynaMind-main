# 文件: EEG2Video/diffusion_prior/model.py

import torch
import torch.nn as nn
import random
from tqdm import tqdm
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists, NoiseScheduler, prob_mask_like
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, \
    rearrange, LayerNorm, RelPosBias, Attention, FeedForward


class BrainDiffusionPrior(DiffusionPrior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip = None
        print("BrainDiffusionPrior: self.clip has been explicitly set to None after initialization.")

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        # 移除 drop_prob 参数，因为它们不再需要
        text_cond_copy = text_cond.copy()
        text_cond_copy.pop('text_cond_drop_prob', None)
        text_cond_copy.pop('image_cond_drop_prob', None)
        text_cond_copy.pop('contra_cond_drop_prob', None)

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            **text_cond_copy
        )

        # ... (后续损失计算保持不变)
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)
        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise
        loss = nn.functional.mse_loss(pred, target)
        return loss, pred

    def forward(
            self,
            text_embed=None,
            image_embed=None,
            image_embed_cond=None,
            contra_embed=None,
            text_encodings=None,
            *args,
            **kwargs
    ):
        assert exists(image_embed), 'image_embed (now target CLIP text embed) must be supplied'

        text_cond = dict(
            text_embed=text_embed,
            image_embed_cond=image_embed_cond,
            contra_embed=contra_embed,
            # 将 drop_prob 设为 0，以确保总是使用传入的嵌入
            text_cond_drop_prob=0.0,
            image_cond_drop_prob=0.0,
            contra_cond_drop_prob=0.0
        )

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        loss, pred = self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)

        return loss, pred


class PriorNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ... (保持不变)

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            text_embed=None,
            image_embed_cond=None,
            contra_embed=None,

            # 移除 drop_prob 参数，因为它们将在 PriorNetwork 的 forward 中被移除
    ):
        batch, seq_len_target, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # --- 核心修改：直接使用 default 逻辑处理 None 值 ---
        # 如果传入的嵌入是 None，则直接使用空嵌入
        text_embed = default(text_embed, lambda: self.null_text_embed.to(dtype).repeat(batch, 1, 1))
        image_embed_cond = default(image_embed_cond, lambda: self.null_image_embed_cond.to(dtype).repeat(batch, 1, 1))
        contra_embed = default(contra_embed, lambda: self.null_contra_embed.to(dtype).repeat(batch, 1, 1))

        # ... (后续的序列拼接和 transformer 调用保持不变)
        tokens = torch.cat((
            text_embed,
            image_embed_cond,
            contra_embed,
            image_embed,
            learned_queries_final
        ), dim=-2)

        tokens = self.causal_transformer(tokens)
        pred_embed = tokens[..., -self.num_tokens:, :]
        return pred_embed