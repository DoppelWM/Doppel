import torch.nn as nn
import torch.nn.functional as F


class IRS(nn.Module):
    """MLP used for the implicit skinning weights and the appearance deformation."""

    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        return self.layers[-1](x)
