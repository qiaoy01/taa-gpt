from torch import nn
from .sdpa import ScaledDotProductAttention
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout=0.1, scale=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()

        residual = q

        # Linear transformations
        q = self.w_qs(q).view(batch_size, self.n_head, seq_len, self.d_k)
        k = self.w_ks(k).view(batch_size, self.n_head, seq_len, self.d_k)
        v = self.w_vs(v).view(batch_size, self.n_head, seq_len, self.d_v)

        # Scaled dot product attention
        scores, attn = self.attention(q, k, v, mask)

        # Reshape scores back to the original shape
        scores = scores.view(batch_size, self.n_head, seq_len, self.d_v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.d_v)

        # Concatenate and apply fully connected layer
        output = self.fc(scores)

        # Restore q to its original shape
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_head * self.d_k)

        return self.layer_norm(output + residual), attn



 
