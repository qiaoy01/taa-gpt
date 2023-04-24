'''
PositionwiseFeedForward 1.0
'''
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, device, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff, bias=True, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=d_ff, out_features=d_model, bias=True, device=device),
            nn.LayerNorm(normalized_shape=d_model, device=device)
        )

    def forward(self, x):
        x = x.to(self.device)
        y = self.fc(x)
        return x + y
    
if __name__ == '__main__':
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
    print("Output shape verification test passed!")
