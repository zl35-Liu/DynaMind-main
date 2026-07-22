import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ptflops import get_model_complexity_info
from omegaconf import OmegaConf
import argparse
from moudules.RSM.models.reign_mapper import UnifiedEEGModel
from moudules.RSM.models.diffusion_prior import PriorNetwork,BrainDiffusionPrior
from moudules.TDA.models.temporal_aligner import EEGVideoAlignmentModel



import math
import random
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from copy import deepcopy
max_length = 16

























































































































































































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
        assert exists(image_embed), 'image_embed (target) must be supplied'

        text_cond = dict(
            text_embed=text_embed,
            image_embed_cond=image_embed_cond,
            contra_embed=contra_embed,
            text_cond_drop_prob=0.0,
            image_cond_drop_prob=0.0,
            contra_cond_drop_prob=0.0
        )

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present'
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





    ):
        batch = 1
        seq_len_target = 77

        dim = 768
        device, dtype = image_embed.device, image_embed.dtype

        time_emb = self.time_embed(torch.tensor([100], dtype=torch.float16))












        text_embed = torch.randn(1,77,768)
        image_embed_cond = torch.randn(1,77,768)
        contra_embed = torch.randn(1,1,768)

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

        tokens = self.causal_transformer(tokens)

        pred_embed = tokens[..., -self.num_tokens:, :]
        return self.to_pred_embed(pred_embed)



model = PriorNetwork(dim = 768, depth=12,heads=64,learned_query_mode='token')
input_res = (1,768)
































































def input_constructor(input_res):


    input1 = torch.randn(62, 100, 7)
    input2 = torch.randn(6,4,36,64)


    return {'x1': input1, 'x2': input2}

macs, params = get_model_complexity_info(
    model,
    input_res,
    as_strings=True,
    print_per_layer_stat=True,
)


print(f"\n--- CVPR Compute Report ---")
print(f"Model: {model.__class__.__name__}")




print(f"Total MACs (Giga Multiply-Accumulate): {macs}")


print(f"Model Parameters: {params}")



macs_float, params_float = get_model_complexity_info(
    model,
    input_res,
    as_strings=False,
    print_per_layer_stat=False,
)


total_flops = macs_float * 2

print(f"\n--- Numerical Results for CVPR CRF ---")
print(f"FLOPs per forward pass (approximate): {total_flops / 10**9:.2f} GFLOPs")
print(f"Model Size (Parameters): {params_float / 10**6:.2f} M")
