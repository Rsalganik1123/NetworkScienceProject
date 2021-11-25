import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Softmax()
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
