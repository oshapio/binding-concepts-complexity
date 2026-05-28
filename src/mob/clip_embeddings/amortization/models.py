import math
from collections import OrderedDict

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pe
        return self.dropout(x)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x):
        return self.resblocks(x)


class SceneEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_out_size: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        max_scene_size: int,
        pad_token: int,
        dropout: float = 0.0,
        use_cliplike_text_encoder: bool = False,
    ):
        super().__init__()
        self.token_embedder = torch.nn.Embedding(vocab_size, d_model)
        self.use_cliplike_text_encoder = use_cliplike_text_encoder
        self.max_scene_size = max_scene_size
        self.pad_token = pad_token

        if use_cliplike_text_encoder:
            self.transformer = Transformer(
                width=d_model,
                layers=num_layers,
                heads=num_heads,
                attn_mask=self.build_attention_mask(),
            )
            self.positional_embedding = nn.Parameter(torch.empty(max_scene_size, d_model))
            self.ln_final = LayerNorm(d_model)
            self.text_projection = nn.Parameter(torch.empty(d_model, d_out_size))
        else:
            self.positional_embedding = PositionalEncoding(d_model, dropout, max_scene_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead=num_heads,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.out_proj = nn.Linear(d_model, d_out_size)

        self._init_weights()

    def _init_weights(self):
        if self.use_cliplike_text_encoder:
            nn.init.normal_(self.token_embedder.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)

            proj_std = (self.transformer.width ** -0.5) * (
                (2 * self.transformer.layers) ** -0.5
            )
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.text_projection is not None:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, seqs_bt):
        if self.use_cliplike_text_encoder:
            x = self.token_embedder(seqs_bt)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            last_non_pad_b = torch.sum((seqs_bt == self.pad_token), dim=1) - 1
            x = x[torch.arange(x.shape[0]), last_non_pad_b] @ self.text_projection
            return x

        vocab_embedded_btd = self.token_embedder(seqs_bt)
        positioned_btd = self.positional_embedding(vocab_embedded_btd)
        padding_mask_btt = seqs_bt == self.pad_token
        transformer_out_btd = self.transformer_encoder(
            positioned_btd, src_key_padding_mask=padding_mask_btt
        )
        last_non_pad_b = torch.sum((padding_mask_btt == 0), dim=1) - 1
        eos_embedding_bd = transformer_out_btd[
            torch.arange(transformer_out_btd.shape[0]), last_non_pad_b
        ]
        return self.out_proj(eos_embedding_bd)

    def build_attention_mask(self):
        mask = torch.empty(self.max_scene_size, self.max_scene_size)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask
