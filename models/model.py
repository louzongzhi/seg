import torch
import torch.nn as nn
import torch.nn.functional as F

class self_net(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        pass

    def forward(self, x):
        pass
        return x
