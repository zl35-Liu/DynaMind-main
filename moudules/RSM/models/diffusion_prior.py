

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
    def __init__(
            self,
            dim,
            depth,
            heads,
            dim_head=64,
            causal=False,
            attn_dropout=0.,
            ff_dropout=0.,
            num_tokens=77,
            learned_query_mode="none",
            text_embed_dim=768,
            image_embed_cond_dim=768,
            contra_embed_dim=512,
            self_cond=False,
    ):
        super().__init__()
        assert learned_query_mode in {'none', 'token', 'mlp'}, 'learned_query_mode must be one of none, token, or mlp'

        self.dim = dim
        self.num_tokens = num_tokens
        self.self_cond = self_cond

        if self.self_cond:
            self.self_cond_to_init_embed = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

        self.learned_query_mode = learned_query_mode

        if learned_query_mode == 'token':
            self.learned_queries = nn.Parameter(torch.randn(num_tokens, dim))
        elif learned_query_mode == 'mlp':
            self.learned_query_mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, num_tokens * dim)
            )
            self.learned_query_layernorm = nn.LayerNorm(dim)

        self.to_text_embed = nn.Linear(text_embed_dim, dim)
        self.to_image_embed_cond = nn.Linear(image_embed_cond_dim, dim)
        self.to_contra_embed = nn.Linear(contra_embed_dim, dim)

        self.null_text_embed = nn.Parameter(torch.randn(1, 1, dim))
        self.null_image_embed_cond = nn.Parameter(torch.randn(1, 1, dim))
        self.null_contra_embed = nn.Parameter(torch.randn(1, 1, dim))

        self.time_embed = SinusoidalPosEmb(dim)

        self.causal_transformer = CausalTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,

            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,

        )

        self.to_pred_embed = nn.Linear(dim, dim)

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            text_embed=None,
            image_embed_cond=None,
            contra_embed=None
    ):
        batch, seq_len_target, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        time_emb = self.time_embed(diffusion_timesteps)

        text_embed = default(text_embed, lambda: self.null_text_embed.to(dtype).repeat(batch, 1, 1))
        image_embed_cond = default(image_embed_cond, lambda: self.null_image_embed_cond.to(dtype).repeat(batch, 1, 1))
        contra_embed = default(contra_embed, lambda: self.null_contra_embed.to(dtype).repeat(batch, 1, 1))

        if text_embed.ndim == 2:
            text_embed = rearrange(text_embed, 'b d -> b 1 d')
        if image_embed_cond.ndim == 2:
            image_embed_cond = rearrange(image_embed_cond, 'b d -> b 1 d')
        if contra_embed.ndim == 2:
            contra_embed = rearrange(contra_embed, 'b d -> b 1 d')

        text_embed = self.to_text_embed(text_embed)
        image_embed_cond = self.to_image_embed_cond(image_embed_cond)
        contra_embed = self.to_contra_embed(contra_embed)

        init_embed = None
        if self.self_cond:
            init_embed = self.self_cond_to_init_embed(self_cond)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_queries, 'n d -> b n d', b=batch)
        elif self.learned_query_mode == 'mlp':
            learned_queries = self.learned_query_mlp(time_emb)
            learned_queries = rearrange(learned_queries, 'b (n d) -> b n d', n=self.num_tokens)
            learned_queries = self.learned_query_layernorm(learned_queries)

        tokens = torch.cat((
            text_embed,
            image_embed_cond,
            contra_embed,
            image_embed,
            learned_queries
        ), dim=-2)

        tokens = self.causal_transformer(tokens, context_time_emb=time_emb, init_embed=init_embed)

        pred_embed = tokens[..., -self.num_tokens:, :]
        return self.to_pred_embed(pred_embed)
