import torch.nn as nn

from mnist_mlp.config import TrainConfig


class MNISTMLP(nn.Module):
    """Three-layer MLP for MNIST classification.

    Input:  28x28 grayscale image  → flattened to 784
    Hidden: 784 → hidden1 → hidden2
    Output: hidden2 → 10 class logits
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, 10),
        )

    def forward(self, x):
        return self.network(x)
