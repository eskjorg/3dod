"""Neural network model."""

from importlib import import_module
from torch import nn
from lib.utils import get_layers

class Model(nn.Module):
    """Neural network module."""

    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self._encoder, self._bottleneck_channels, downsampling = self._create_encoder()
        self._decoder = self._create_decoder(downsampling)
        if self._configs.training.nll_loss:
            self._decoder_ln_b = self._create_decoder(downsampling)
        self._encoder_output_channels = None

    def _create_encoder(self):
        encoder_name = self._configs.network.encoder
        module = import_module('lib.models.{}'.format(encoder_name))
        return module.get_encoder()

    def _create_decoder(self, downsampling):
        return MultiTaskNet(get_layers(self._configs.config_name),
                            in_channels=self._bottleneck_channels,
                            upsampling_factor=int(downsampling / self._configs.network.output_stride))

    def forward(self, input_data):
        features = self._encoder(input_data)
        outputs_task = self._decoder(features)
        outputs_ln_b = self._decoder_ln_b(features) if self._configs.training.nll_loss else {}
        return outputs_task, outputs_ln_b


class MultiTaskNet(nn.ModuleDict):
    """MultiTaskNet."""
    def __init__(self, layers, in_channels, upsampling_factor):
        heads = {}
        for name, settings in layers.items():
            heads[name] = MultiTaskHead(in_channels=in_channels,
                                        out_channels=settings['n_layers'],
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
                   nn.Dropout2d(p=0.5, inplace=False),
                   nn.Conv2d(in_channels=in_channels * 2,
                             out_channels=out_channels * upsampling ** 2,
                             kernel_size=1,
                             stride=1)]
        if upsampling != 1:
            modules.append(nn.PixelShuffle(upsampling))
        super(MultiTaskHead, self).__init__(*modules)
