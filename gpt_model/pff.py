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
    
