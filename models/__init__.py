from .unet import UNet
from .model import self_net
from .vmunet.vmunet import VMUNet

__all__ = [
    'UNet',
    'self_net',
    'VMUNet',
]


def load_model(model_name):
    if model_name == 'UNet':
        return UNet(n_channels=3, n_classes=4, bilinear=False)
    elif model_name == 'self_net':
        return self_net(n_channels=3, n_classes=4)
    elif model_name == 'VMUNet':
        return VMUNet()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
