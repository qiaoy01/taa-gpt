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



def test_scaled_dot_product_attention():
    attention = ScaledDotProductAttention(dropout=0.1, scale=True)

    q = torch.randn(8, 3, 64, 64, requires_grad=True)  # batch_size=8, n_head=3, seq_len=64, d_k=64
    k = torch.randn(8, 3, 64, 64, requires_grad=True)
    v = torch.randn(8, 3, 64, 64, requires_grad=True)

    mask = torch.ones(8, 3, 64, 64).bool()
    mask[:, :, 2, 2] = 0

    output, attn = attention(q, k, v, mask)

    assert output.shape == (8, 3, 64, 64), "Output shape mismatch"
    assert attn.shape == (8, 3, 64, 64), "Attention shape mismatch"

    assert torch.all(output >= -1e9) and torch.all(output <= 1e9), "Output values out of range"
    assert torch.all(torch.isclose(attn, attn, atol=1e-6)), "Attention values out of range"

    output.sum().backward()
    assert q.grad is not None, "Gradient not propagating to q"
    assert k.grad is not None, "Gradient not propagating to k"
    assert v.grad is not None, "Gradient not propagating to v"

    print("All tests passed.")

if __name__ == '__main__':
    test_scaled_dot_product_attention()

