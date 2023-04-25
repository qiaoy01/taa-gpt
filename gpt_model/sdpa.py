import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=-1)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.einsum('bnqd,bnkd->bnqk', q, k)

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        
        output = torch.einsum('bnqk,bnkd->bnqd', attn, v)
        return output, attn



