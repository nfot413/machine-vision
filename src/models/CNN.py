import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, dropout_p: float = 0.25):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)   # -> (B, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)            # -> (B, 64, 28, 28)
        self.pool  = nn.MaxPool2d(2, 2)                                     # 尺寸减半
        self.dropout = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))                 # (B, 32, 28, 28)
        x = F.relu(self.conv2(x))                 # (B, 64, 28, 28)
        x = self.pool(x)                          # (B, 64, 14, 14)
        x = self.pool(x)                          # (B, 64, 7, 7)

        x = self.dropout(x)
        x = torch.flatten(x, 1)                   # (B, 64*7*7)
        x = F.relu(self.fc1(x))                   # (B, 128)
        x = self.fc2(x)                           # (B, num_classes)
        return x