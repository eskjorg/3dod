"""Neural network model."""

from torch import nn
from lib import models
from lib.utils import get_layers

class Model(nn.Module):
    """Neural network module."""

    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()

    def _create_encoder(self):
        return getattr(models, self._configs.encoder).get_module()

    def _create_decoder(self):
        return MultiTaskNet(get_layers(self._configs.config_load_path),
                            in_channels=self._encoder.out_channels_backbone,
                            upsampling_factor=self._configs.upsampling_factor)

    def forward(self, input_data):
        return self._encoder(self._decoder(input_data))


class MultiTaskNet(nn.ModuleDict):
    """MultiTaskNet."""
    def __init__(self, layers, in_channels, upsampling_factor):
        heads = {}
        for name, settings in layers.items():
            heads[name] = MultiTaskHead(in_channels=in_channels,
                                        out_channels=settings.n_layers,
                                        upsampling=upsampling_factor)
        super(MultiTaskNet, self).__init__(heads)

    def forward(self, x):
        """Forward pass of input through module."""
        return {task_name: task_head(x) for task_name, task_head in self.items()}


class MultiTaskHead(nn.Sequential):
    """MultiTaskHead."""
    def __init__(self, in_channels, out_channels, upsampling=1):
        modules = [nn.Conv2d(in_channels=in_channels,
                             out_channels=in_channels * 2,
                             kernel_size=1,
                             stride=1),
                   nn.BatchNorm2d(num_features=in_channels * 2),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels=in_channels * 2,
                             out_channels=out_channels * upsampling ** 2,
                             kernel_size=1,
                             stride=1)]
        if upsampling != 1:
            modules.append(nn.PixelShuffle(upsampling))
        super(MultiTaskHead, self).__init__(*modules)
