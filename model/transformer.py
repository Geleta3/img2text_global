import torch
from torch import nn
import torch.nn.functional as F
import math
from torch import Tensor

#
# The structure:
# attention-function +
# Positional - Encoding +
# # Embedding - class
# Multi-head attention
# Feed-forward
# # Norm
# # Dropout
# Decoder Layer
#

torch.manual_seed(2809)


def attention(q, k, v, heads, trg_mask=None):
    bs, seq_len, d_model = q.size()
    assert d_model % heads == 0, "dim of model should be divisible by head"
    dk = d_model // heads
    q = q.view(bs, -1, heads, dk).transpose(2, 1)
    k = k.view(bs, -1, heads, dk).transpose(2, 1)
    v = v.view(bs, -1, heads, dk).transpose(2, 1)

    att_weight = q.matmul(k.transpose(-1, -2))
    if trg_mask is not None:
        # Eg. trg_mask = [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        att_weight = att_weight.masked_fill(trg_mask.unsqueeze(1) == 0, -1e9)
    att_weight = F.softmax(att_weight / math.sqrt(dk), dim=-1)
    return att_weight.matmul(v)


class PositionalEncoding:
    def __init__(self, seq_len, d_model, device="cpu"):
        self.d_model = d_model
        pos = torch.arange(seq_len).view(1, -1, 1).to(device)
        d = torch.arange(d_model).view(1, 1, d_model).to(device)
        phase = torch.div(pos, (torch.pow(1e4, torch.div(d, d_model))))
        self.pos_enc = torch.where(d % 2 == 0, torch.sin(phase), torch.cos(phase))

    def __call__(self, emb_out: Tensor):
        bs, seq, d_model = emb_out.size()
        emb_out = emb_out * math.sqrt(self.d_model)
        return emb_out + self.pos_enc[:, :seq]


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, src_mask=None, trg_mask=None):
        temp = q
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        x = attention(q=q, k=k, v=v,
                      heads=self.heads,
                      trg_mask=trg_mask)
        x = torch.cat([x[:, _, :, :] for _ in range(self.heads)], dim=-1)
        out = self.norm(x + temp)
        return out


class FeedForward(nn.Module):
    def __init__(self, feed_dim, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, feed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(feed_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        temp = x
        x = self.dropout(self.linear1(x))
        x = self.linear2(self.relu(x))
        return self.norm(x + temp)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, feed_dim, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mul_att = MultiHeadAttention(heads=heads,
                                          d_model=d_model,
                                          dropout=dropout)

        self.feed_forward = FeedForward(feed_dim=feed_dim,
                                        d_model=d_model,
                                        dropout=dropout)

    def forward(self, src, src_mask=None):
        x = self.mul_att(src, src, src, src_mask=src_mask)
        # print("Encoder: ", x[:, 0:4, 10:20])
        x = self.feed_forward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, feed_dim, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mul_att1 = MultiHeadAttention(heads=heads,
                                           d_model=d_model,
                                           dropout=dropout)
        self.mul_att2 = MultiHeadAttention(heads=heads,
                                           d_model=d_model,
                                           dropout=dropout)

        self.feed_forward = FeedForward(feed_dim=feed_dim,
                                        d_model=d_model,
                                        dropout=dropout)

    def forward(self, memory, trg, src_mask=None, trg_mask=None):
        x = self.mul_att1(trg, trg, trg, trg_mask=trg_mask)
        x = self.mul_att2(q=x, k=memory, v=memory, )
        x = self.feed_forward(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, heads, feed_dim, n_layer,
                 max_seq=64, dropout=0.1, device="cpu"):
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model,
                                                      seq_len=max_seq,
                                                      device=device)

        self.encoder = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                   feed_dim=feed_dim,
                                                   heads=heads,
                                                   dropout=dropout) for _ in range(n_layer)])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = self.positional_encoding(src)
        x = self.norm(x)
        # x = src
        for layer in self.encoder:
            x = layer(x, src_mask=src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, heads, feed_dim, n_layer,
                 vocab_size, max_seq=60, dropout=0.1, device="cpu"):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model,
                                                      seq_len=max_seq,
                                                      device=device)

        self.decoder = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                   feed_dim=feed_dim,
                                                   heads=heads,
                                                   dropout=dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.last_layer = nn.Linear(d_model, vocab_size)

    def forward(self, memory, trg, trg_mask):
        x = self.embedding(trg)
        x = self.positional_encoding(x)
        x = self.norm(x)
        for layer in self.decoder:
            x = layer(memory=memory, trg=x, trg_mask=trg_mask)
        x = self.last_layer(x)
        return x



