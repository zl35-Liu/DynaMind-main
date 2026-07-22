import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Conv2dBlock(nn.Module):
    """
     Conv2d -> BN -> GELU -> Dropout
    """
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=None, drop=0.0):
        super().__init__()
        if padding is None:

            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class EEGToTextGlobalEncoder(nn.Module):

    """
    Encoder using 2D Conv for feature extraction, then MLP for global projection.
    EEG (B, 64, 4000) -> Flatten -> MLP -> (B, 226, 4096)
    """
    def __init__(
        self,
        in_channels=64,
        target_seq_len=50,
        target_embed_dim=4096,
        conv_output_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_embed_dim = target_embed_dim


        self.target_flat_dim = target_seq_len * target_embed_dim



        self.conv_layers = nn.Sequential(

            Conv2dBlock(1, 16, kernel_size=(3, 9), stride=(2, 2), drop=dropout),

            Conv2dBlock(16, 32, kernel_size=(3, 5), stride=(2, 2), drop=dropout),

            Conv2dBlock(32, conv_output_dim, kernel_size=(3, 3), drop=dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))



        with torch.no_grad():
            dummy_input = torch.randn(1, 1, in_channels, 4000)
            dummy_output = self.conv_layers(dummy_input)
            self.flat_dim = dummy_output.numel() // 1
            print(f"2D Conv Output Shape: {dummy_output.shape}. MLP Input Dim: {self.flat_dim}")
            pool_output = self.global_pool(dummy_output).view(1, -1)
            print(f"After Global Pooling Shape: {pool_output.shape}")



        self.mapper_mlp = nn.Sequential(
            nn.Linear(conv_output_dim, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, self.target_flat_dim)
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                pass

    def forward(self, x):
        """
        x: (B, 64, 4000)
        returns: (B, target_seq_len=226, target_embed_dim=4096)
        """
        B = x.shape[0]


        x_2d = x.unsqueeze(1)


        conv_features = self.conv_layers(x_2d)
        pool_features = self.global_pool(conv_features).view(B, -1)





        flat_emb = self.mapper_mlp(pool_features)


        final_emb = flat_emb.view(B, self.target_seq_len, self.target_embed_dim)

        return final_emb

class EEGToTextGlobalEncoder1(nn.Module):
    """
    Encoder variant designed to reduce information loss caused by pooling.
    """
    def __init__(
        self,
        in_channels=64,
        target_seq_len=50,
        target_embed_dim=4096,
        conv_output_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_embed_dim = target_embed_dim
        self.target_flat_dim = target_seq_len * target_embed_dim



        self.conv_layers = nn.Sequential(



            Conv2dBlock(1, 16, kernel_size=(5, 17), stride=(8, 8), drop=dropout),



            Conv2dBlock(16, 32, kernel_size=(3, 9), stride=(8, 8), drop=dropout),



            Conv2dBlock(32, conv_output_dim, kernel_size=(1, 9), stride=(1, 8), drop=dropout),
        )







        with torch.no_grad():
            dummy_input = torch.randn(1, 1, in_channels, 4000)
            dummy_output = self.conv_layers(dummy_input)
            self.flat_dim_conv = dummy_output.numel() // 1

            print(f"2D Conv Output Shape: {dummy_output.shape}. MLP Input Dim: {self.flat_dim_conv}")



        self.mapper_mlp = nn.Sequential(
            nn.Linear(self.flat_dim_conv, 4096),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4096, self.target_flat_dim)
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                pass

    def forward(self, x):
        """
        x: (B, 64, 4000)
        returns: (B, target_seq_len=50, target_embed_dim=4096)
        """
        B = x.shape[0]
        x_2d = x.unsqueeze(1)


        conv_features = self.conv_layers(x_2d)



        flat_features = torch.flatten(conv_features, start_dim=1)


        flat_emb = self.mapper_mlp(flat_features)


        final_emb = flat_emb.view(B, self.target_seq_len, self.target_embed_dim)

        return final_emb


class Conv1dBlock(nn.Module):
    """
     Conv1d -> BN -> GELU -> Dropout (optional)
    """
    def __init__(self, in_ch, out_ch, kernel_size=9, stride=1, padding=None, drop=0.0):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class ResidualConvBlock(nn.Module):
    """
     Simple residual block for 1D convs
    """
    def __init__(self, channels, kernel_size=9, dropout=0.0):
        super().__init__()
        self.conv1 = Conv1dBlock(channels, channels, kernel_size=kernel_size, drop=dropout)
        self.conv2 = Conv1dBlock(channels, channels, kernel_size=kernel_size, drop=dropout)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class EEGToTextEncoder(nn.Module):
    """
    Convert EEG (B, 64, 4000) -> (B, target_seq_len=226, target_embed_dim=4096)

    Configurable via kwargs:
      - initial_channels: convolutional stem channels (e.g. 128/256)
      - mid_channels: channels before final expansion
      - target_seq_len: 226
      - target_embed_dim: 4096
      - n_res_blocks: number of residual conv blocks
      - use_transformer: whether to run Transformer encoder on tokens
      - transformer_layers, transformer_heads: transformer config
    """
    def __init__(
        self,
        in_channels=64,
        initial_channels=128,
        mid_channels=1024,
        target_seq_len=226,
        target_embed_dim=4096,
        n_res_blocks=2,
        use_transformer=True,
        transformer_layers=2,
        transformer_heads=16,
        dropout=0.1,
        eps=1e-6
    ):
        super().__init__()
        self.in_channels = in_channels
        self.target_seq_len = target_seq_len
        self.target_embed_dim = target_embed_dim


        self.stem = nn.Sequential(
            Conv1dBlock(in_channels, initial_channels, kernel_size=9, drop=dropout),
            Conv1dBlock(initial_channels, initial_channels, kernel_size=9, drop=dropout)
        )


        self.res_blocks = nn.Sequential(*[ResidualConvBlock(initial_channels, kernel_size=9, dropout=dropout)
                                         for _ in range(n_res_blocks)])


        self.expand = nn.Conv1d(initial_channels, mid_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm1d(mid_channels)
        self.expand_act = nn.GELU()


        self.pool = nn.AdaptiveAvgPool1d(target_seq_len)




        proj_dim = min(target_embed_dim, 2048)
        self.project1 = nn.Conv1d(mid_channels, proj_dim, kernel_size=1, bias=False)
        self.project1_bn = nn.BatchNorm1d(proj_dim)
        self.project1_act = nn.GELU()
        self.project2 = nn.Conv1d(proj_dim, target_embed_dim, kernel_size=1, bias=True)


        self.use_transformer = use_transformer
        if use_transformer:

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=target_embed_dim,
                nhead=transformer_heads,
                dim_feedforward=target_embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

            self.pre_ln = nn.LayerNorm(target_embed_dim, eps=eps)
            self.post_ln = nn.LayerNorm(target_embed_dim, eps=eps)
        else:
            self.transformer = None
            self.pre_ln = nn.LayerNorm(target_embed_dim, eps=eps)


        self.output_dropout = nn.Dropout(dropout)


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):

                pass

    def forward(self, x):
        """
        x: (B, 64, 4000)
        returns: (B, target_seq_len=226, target_embed_dim=4096)
        """


        x = self.stem(x)
        x = self.res_blocks(x)


        x = self.expand(x)
        x = self.expand_bn(x)
        x = self.expand_act(x)


        x = self.pool(x)


        x = self.project1(x)
        x = self.project1_bn(x)
        x = self.project1_act(x)
        x = self.project2(x)


        x = x.transpose(1, 2)
        x = self.pre_ln(x)
        if self.transformer is not None:

            x_t = x.transpose(0, 1)
            x_t = self.transformer(x_t)
            x = x_t.transpose(0, 1)
            x = self.post_ln(x)
        x = self.output_dropout(x)
        return x


class UnifiedEEGModelForSeqText(nn.Module):
    """
    Unified model where shared_encoder yields sequence text-like embeddings:
      - shared_features: (B, seq_len, embed_dim)
      - text_emb: (B, seq_len, embed_dim)  (normalized and scaled)
      - image_emb / classifiers unchanged (optional)
    """
    def __init__(self, cfg, encoder_cfg=None):
        super().__init__()

        if encoder_cfg is None:
            encoder_cfg = {}
        self.shared_encoder = EEGToTextGlobalEncoder1(
            in_channels=encoder_cfg.get("in_channels", 64),
            target_seq_len=encoder_cfg.get("target_seq_len", 50),
            target_embed_dim=encoder_cfg.get("target_embed_dim", 4096),
            conv_output_dim=encoder_cfg.get("conv_output_dim", 1024),
            dropout=encoder_cfg.get("dropout", 0.1)
        )
        print(f"Initialized EEGToTextGlobalEncoder with target_seq_len={self.shared_encoder.target_seq_len}")














        self.cfg = cfg

        self.scale = nn.Parameter(torch.tensor(1.0))



        if cfg.training_tasks.text_alignment.enabled:


            per_token_dim = encoder_cfg.get("target_embed_dim", 4096)
            self.text_proj = nn.Sequential(
                nn.Linear(per_token_dim, per_token_dim),
                nn.LayerNorm(per_token_dim)
            )
        else:
            self.text_proj = None



        self.image_projection = None
        if cfg.training_tasks.image_alignment.enabled:

            self.image_projection = nn.Sequential(
                nn.Linear(encoder_cfg.get("target_embed_dim", 4096), cfg.training_tasks.image_alignment.emb_dim),
                nn.LayerNorm(cfg.training_tasks.image_alignment.emb_dim)
            )

        if cfg.training_tasks.classification.enabled:

            self.classifiers = nn.ModuleDict({
                task_name: nn.Sequential(
                    nn.Linear(encoder_cfg.get("target_embed_dim", 4096), 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                ) for task_name, num_classes in cfg.training_tasks.classification.tasks.items()
            })
        else:
            self.classifiers = None

    def forward(self, x):
        """
        x: (B, 64, 4000)
        returns:
          - shared_features: (B, seq_len=226, embed_dim=4096)
          - text_emb: normalized (B, seq_len, embed_dim)
          - image_emb: (B, emb_dim) or None
          - cls_logits: dict or None
        """
        shared_features = self.shared_encoder(x)

        text_emb = None
        image_emb = None
        cls_logits = None

        if self.text_proj is not None:

            B, S, E = shared_features.shape
            text = self.text_proj(shared_features)

            text = F.normalize(text, dim=-1) * self.scale
            text_emb = text

        if self.image_projection is not None:

            pooled = shared_features.mean(dim=1)
            image_emb = self.image_projection(pooled)
            image_emb = F.normalize(image_emb, dim=-1) * self.scale

        if self.classifiers is not None:
            pooled = shared_features.mean(dim=1)
            cls_logits = {tn: clsr(pooled) for tn, clsr in self.classifiers.items()}

        return shared_features, text_emb, image_emb, cls_logits


if __name__ == "__main__":

    class DummyCfg:
        class _TA:
            enabled = True
            emb_dim = 226 * 4096
        class _IA:
            enabled = False
            emb_dim = 768
        class _CL:
            enabled = False
            tasks = {}
        training_tasks = type("T", (), {"text_alignment": _TA, "image_alignment": _IA, "classification": _CL})

    cfg = DummyCfg()

    encoder_cfg = {"in_channels": 64, "initial_channels": 128, "mid_channels": 1024,
                   "target_seq_len": 226, "target_embed_dim": 4096, "use_transformer": True}
    model = UnifiedEEGModelForSeqText(cfg, encoder_cfg=encoder_cfg)


    x = torch.randn(2, 64, 4000)
    shared, text_emb, image_emb, cls = model(x)
    print("shared:", shared.shape)
    print("text_emb:", None if text_emb is None else text_emb.shape)
    if text_emb is not None:

        print("per-token norm example:", text_emb[0,0,:].norm().item())
