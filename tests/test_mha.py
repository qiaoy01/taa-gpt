import torch
from gpt_model.pff import PositionwiseFeedForward
from gpt_model.sdpa import ScaledDotProductAttention
from gpt_model.mha import MultiHeadAttention

def test_multihead_attention():
    n_head = 4
    d_model = 64
    d_k = 16
    d_v = 16
    seq_len = 8
    batch_size = 2

    multihead_attention = MultiHeadAttention(n_head, d_model, d_k, d_v)

    q = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    k = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    v = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    mask = None

    output, attn = multihead_attention(q, k, v, mask)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    assert attn.shape == (batch_size, n_head, seq_len, seq_len), "Attention shape mismatch"

    # Check gradient propagation
    output.sum().backward()
    assert q.grad is not None, "Gradient not propagating to q"
    assert k.grad is not None, "Gradient not propagating to k"
    assert v.grad is not None, "Gradient not propagating to v"

    print("MultiHeadAttention tests passed.")

def test_pff():
    batch_size = 2
    seq_len = 3
    d_model = 4
    d_ff = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, d_model)
    model = PositionwiseFeedForward(d_model, d_ff, device=device, dropout=0.1)
    expected_output_shape = torch.Size([batch_size, seq_len, d_model])
    output = model(x)
    assert output.shape == expected_output_shape
    print("PositionwiseFeedForward Output shape verification test passed!")

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

    print("ScaledDotProductAttention tests passed.")

if __name__ == "__main__":
    test_pff()
    test_scaled_dot_product_attention()
    test_multihead_attention()
