import torch
import torch.nn as nn
from pff import PositionwiseFeedForward
from mha import MultiHeadAttention

class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, device, dropout=0.1):
        super(GPTDecoderLayer, self).__init__()
        self.device = device
        #self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, device=device, dropout=dropout)
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, d_k=self.d_k, d_v=self.d_v,dropout=0.1).to(device)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, device=device)

        self.layer_norm1 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x.to(self.device)
        # Self-attention block
        _x, _ = self.self_attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(_x))

        # Position-wise feed-forward block
        _x = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(_x))

        return x


def test_gpt_decoder_layer():
    d_model = 256
    n_head = 8
    d_ff = 1024
    device = torch.device("cpu")

    decoder_layer = GPTDecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, device=device)

    x = torch.randn(4, 16, d_model)
    mask = torch.ones(4, 1, 16, dtype=torch.bool)

    out = decoder_layer(x, mask)

    assert out.shape == (4, 16, d_model), "Output shape doesn't match expected shape"

    print("All tests passed.")


if __name__ == '__main__':
    test_gpt_decoder_layer()


