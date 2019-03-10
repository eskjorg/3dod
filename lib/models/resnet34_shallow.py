"""ResNet 34 (shallow)."""
from torch import nn
from torchvision import models

# class ConvBatchRelu(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         modules = [
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 padding=kernel_size//2,
#                 stride=stride,
#             ),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.ReLU(inplace=True),
#             # nn.Dropout2d(p=0.5, inplace=False),
#         ]
#         super(ConvBatchRelu, self).__init__(*modules)
# 
# def get_encoder():
#     network = models.resnet34(pretrained=True)
# 
#     modules = [
#         ConvBatchRelu(  3,  16, 7, 2),
#         ConvBatchRelu( 16,  32, 3, 1),
#         ConvBatchRelu( 32,  64, 3, 2),
#         ConvBatchRelu( 64,  64, 3, 1),
#         ConvBatchRelu( 64, 128, 3, 2),
#         ConvBatchRelu(128, 128, 3, 1),
#         ConvBatchRelu(128, 128, 1, 1),
#         ConvBatchRelu(128, 128, 3, 1),
#         ConvBatchRelu(128, 128, 1, 1),
#     ]
#     network = nn.Sequential(*modules)
# 
#     downsampling_factor = 8
#     encoder_ch_out = 128
#     return network, encoder_ch_out, downsampling_factor

def get_encoder():
    network = models.resnet34(pretrained=True)

    resnet_lowlevel = nn.Sequential(*list(network.children())[:5])
    resnet_ch_out = list(resnet_lowlevel.children())[-1][-1].bn2.num_features
    encoder_ch_out = 256
    custom = nn.Sequential(*[
        nn.Conv2d(
            in_channels=resnet_ch_out,
            out_channels=encoder_ch_out,
            kernel_size=7,
            padding=3,
            stride=2,
        ),
        nn.BatchNorm2d(num_features=encoder_ch_out),
        nn.ReLU(inplace=True),
    ])
    network = nn.Sequential(resnet_lowlevel, custom)

    downsampling_factor = 8
    return network, encoder_ch_out, downsampling_factor

# def get_encoder():
#     network = models.resnet34(pretrained=True)
# 
#     resnet_lowlevel = nn.Sequential(*list(network.children())[:6])
#     resnet_ch_out = list(resnet_lowlevel.children())[-1][-1].bn2.num_features
#     encoder_ch_out = 128
#     custom = nn.Sequential(*[
#         nn.Conv2d(
#             in_channels=resnet_ch_out,
#             out_channels=encoder_ch_out,
#             kernel_size=7,
#             padding=3,
#             stride=1,
#         ),
#         nn.BatchNorm2d(num_features=encoder_ch_out),
#         nn.ReLU(inplace=True),
#     ])
#     network = nn.Sequential(resnet_lowlevel, custom)
# 
#     downsampling_factor = 8
#     return network, encoder_ch_out, downsampling_factor
