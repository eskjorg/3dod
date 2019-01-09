"""ResNet 34."""
from torch import nn
from torchvision import models

def get_module():
    network = models.resnet34(pretrained=True)
    # Remove fc-layers
    network = nn.Sequential(*list(network.children())[:-2])
    return network
