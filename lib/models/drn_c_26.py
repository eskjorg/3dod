from torch import nn
from third_party import drn

def get_encoder():
    network = getattr(drn, "drn_c_26")(pretrained=True)
    bottleneck_channels = network.fc.in_channels
    network = nn.Sequential(*list(network.children())[:-2])
    downsampling_factor = 8
    return network, bottleneck_channels, downsampling_factor
