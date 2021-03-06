"""Neural network model."""

from importlib import import_module
import torch
from torch import nn
from lib.utils import get_layers, get_device

class Model(nn.Module):
    """Neural network module."""

    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self.encoder, n_bottleneck_channels, downsampling = self._create_encoder()
        self.decoder = self._create_decoder(n_bottleneck_channels, downsampling)

    def _create_encoder(self):
        encoder_name = self._configs.network.encoder
        module = import_module('lib.models.{}'.format(encoder_name))
        return module.get_encoder()

    def _create_decoder(self, n_bottleneck_channels, downsampling):
        return MultiTaskNet(get_layers(self._configs.config_name),
                            in_channels=n_bottleneck_channels,
                            upsampling=int(downsampling / self._configs.network.output_stride),
                            weighting_mode=self._configs.training.weighting_mode)

    def forward(self, input_data):
        features = self.encoder(input_data)
        return self.decoder(features)


class MultiTaskNet(nn.ModuleDict):
    """MultiTaskNet."""
    def __init__(self, layers, in_channels, upsampling, weighting_mode):
        heads = {}
        for name, settings in layers.items():
            # The actual head
            task_head = MultiTaskHead(in_channels=in_channels,
                                      out_channels=settings['n_layers'],
                                      upsampling=upsampling)
            # The loss weighting for that head
            if settings['loss'] == "CE" or weighting_mode == 'uniform':  # no weighting
                task_weighting = ZeroHead()
            elif weighting_mode == 'layer_wise':  # a.k.a. homoscedatic
                task_weighting = LayerWeights(settings['n_layers'])
            elif weighting_mode == 'sample_wise':  # a.k.a. heteroscedastic
                task_weighting = MultiTaskHead(in_channels=in_channels,
                                               out_channels=settings['n_layers'],
                                               upsampling=upsampling)
            else:
                raise NotImplementedError(weighting_mode)
            heads[name] = WeightedHead([task_head, task_weighting])
        super(MultiTaskNet, self).__init__(heads)

    def forward(self, x):
        """Forward pass of input through module."""
        return {task_name: weighted_task_head(x) for task_name, weighted_task_head in self.items()}


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


class WeightedHead(nn.ModuleList):
    def forward(self, x):
        tensor = self[0](x)
        ln_var = self[1](x) * 0.1  # Scale for slower learning rate
        return tensor, ln_var


class LayerWeights(nn.Module):
    def __init__(self, n_weights):
        super().__init__()
        self.register_parameter('weighting', nn.Parameter(torch.zeros(n_weights)))

    def forward(self, x):
        # Scale for increased learning rate
        return 400 * self.weighting

class ZeroHead(nn.Module):
    def forward(self, x):
        return torch.Tensor([0]).to(get_device(), non_blocking=True)