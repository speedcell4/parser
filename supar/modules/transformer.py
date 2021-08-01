# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer


class SinusoidPositionalEmbedding(nn.Module):

    def forward(self, x):
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len)).unsqueeze(-1) / 10000 ** (x.new_tensor(range(n_model)) // 2 * 2 / n_model)
        pos[:, 0::2], pos[:, 1::2] = pos[:, 0::2].sin(), pos[:, 1::2].cos()
        return pos


class SinusoidRelativePositionalEmbedding(nn.Module):

    def forward(self, x):
        seq_len, n_model = x[0].shape
        pos = x.new_tensor(range(seq_len))
        pos = (pos - pos.unsqueeze(-1)).unsqueeze(-1) / 10000 ** (x.new_tensor(range(n_model)) // 2 * 2 / n_model)
        pos[..., 0::2], pos[..., 1::2] = pos[..., 0::2].sin(), pos[..., 1::2].cos()
        return pos


class TransformerEncoder(nn.Module):

    def __init__(self, n_layers, n_heads=8, n_model=1024, n_inner=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model=n_model,
                                                             nhead=n_heads,
                                                             dim_feedforward=n_inner,
                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        x += self.pos_embed(x)
        x, src_key_padding_mask = self.dropout(x).transpose(0, 1), ~mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x.transpose(0, 1)


class RelativePositionTransformerEncoder(nn.Module):

    def __init__(self, n_layers, n_heads=8, n_model=1024, n_inner=2048, pre_norm=True, dropout=0.1):
        super(RelativePositionTransformerEncoder, self).__init__()

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_model = n_model
        self.n_inner = n_inner
        self.pre_norm = pre_norm

        self.layers = nn.ModuleList([RelativePositionTransformerEncoderLayer(n_heads=n_heads,
                                                                             n_model=n_model,
                                                                             n_inner=n_inner,
                                                                             pre_norm=pre_norm,
                                                                             dropout=dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_model) if self.pre_norm else None
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_layers}, {self.n_heads}, n_model={self.n_model}, n_inner={self.n_inner}"
        if self.pre_norm:
            s += f", pre_norm={self.pre_norm}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'
        return s

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        if self.pre_norm:
            x = self.norm(x)
        return x


class RelativePositionMultiHeadAttention(nn.Module):

    def __init__(self, n_heads, n_model, n_embed, dropout=0.1):
        super(RelativePositionMultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.n_model = n_model
        self.n_embed = n_embed
        self.scale = n_embed**0.5

        self.pos_embed = SinusoidRelativePositionalEmbedding()
        self.wq = nn.Parameter(torch.zeros(n_model, n_embed, n_heads))
        self.wk = nn.Parameter(torch.zeros(n_model, n_embed, n_heads))
        self.wv = nn.Parameter(torch.zeros(n_model, n_embed, n_heads))
        self.bu = nn.Parameter(torch.zeros(n_embed, n_heads))
        self.bv = nn.Parameter(torch.zeros(n_embed, n_heads))
        self.wo = nn.Parameter(torch.zeros(n_embed, n_heads, n_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # [batch_size, seq_len, n_embed, n_heads]
        q = torch.einsum('btm,meh->bteh', q, self.wq)
        # [batch_size, seq_len, n_embed, n_heads]
        k = torch.einsum('btm,meh->bteh', k, self.wk)
        # [batch_size, seq_len, n_embed, n_heads]
        v = torch.einsum('btm,meh->bteh', v, self.wv)
        # [seq_len, seq_len, n_embed]
        p = self.pos_embed(q[..., 0])

        attn = torch.einsum('bqeh,bkeh->bqkh', q + self.bu, k) + torch.einsum('bqeh,qke->bqkh', q + self.bv, p)
        attn = attn / self.scale
        attn = attn.masked_fill_(~mask.unsqueeze(-1).repeat(1, 1, self.n_heads).unsqueeze(1), float('-inf')).softmax(-2)
        # [batch_size, seq_len, n_embed, n_heads]
        x = torch.einsum('bqkh,bkeh->bqeh', self.dropout(attn), v)
        # [batch_size, seq_len, n_model]
        x = torch.einsum('bqeh,ehm->bqm', x, self.wo)

        return x


class RelativePositionTransformerEncoderLayer(nn.Module):

    def __init__(self, n_heads, n_model, n_inner, activation='relu', pre_norm=True, dropout=0.1):
        super(RelativePositionTransformerEncoderLayer, self).__init__()

        self.pre_norm = pre_norm

        self.attn = RelativePositionMultiHeadAttention(n_heads, n_model, n_model//8, dropout)
        self.attn_norm = nn.LayerNorm(n_model)
        self.ffn = nn.Sequential(
            nn.Linear(n_model, n_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_inner, n_model)
        )
        self.ffn_norm = nn.LayerNorm(n_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        if self.pre_norm:
            y = self.attn_norm(x)
            x = x + self.dropout(self.attn(y, y, y, mask))
            y = self.ffn_norm(x)
            x = x + self.dropout(self.ffn(y))
        else:
            x = self.attn_norm(x + self.dropout(self.attn(x, x, x, mask)))
            x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x
