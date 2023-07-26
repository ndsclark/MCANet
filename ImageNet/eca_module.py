from torch import nn
import math

__all__ = ['ECALayer']


class ECALayer(nn.Module):
    def __init__(self, inp):
        super(ECALayer, self).__init__()

        gamma = 2
        b = 1
        t = int(abs((math.log2(inp) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return x * out.expand_as(x)
