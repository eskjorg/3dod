"""ResNet 34."""
from torch import nn
from torchvision import models

def get_encoder():
    network = models.resnet34(pretrained=True)
    # Remove fc-layers
    network = nn.Sequential(*list(network.children())[:-2])
    bottleneck_channels = list(network.children())[-1][-1].bn2.num_features
    return network, bottleneck_channels
