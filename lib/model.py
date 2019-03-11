"""Neural network model."""

from importlib import import_module
from torch import nn, Tensor
from lib.utils import get_layers, get_class_map

class Model(nn.Module):
    """Neural network module."""

    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self._class_map = get_class_map(configs)
        self.encoder, n_bottleneck_channels, downsampling = self._create_encoder()
        self.decoder = self._create_decoder(n_bottleneck_channels, downsampling)

    def _create_encoder(self):
        encoder_name = self._configs.network.encoder
        module = import_module('lib.models.{}'.format(encoder_name))
        return module.get_encoder()

    def _create_decoder(self, n_bottleneck_channels, downsampling):
        return MultiTaskNet(self._class_map,
                            get_layers(self._configs.config_name),
                            in_channels=n_bottleneck_channels,
                            upsampling=int(downsampling / self._configs.network.output_stride),
                            weighting_mode=self._configs.training.weighting_mode)

    def forward(self, input_data):
        features = self.encoder(input_data)
        return self.decoder(features)


class MultiTaskNet(nn.ModuleDict):
    """MultiTaskNet."""
    def __init__(self, class_map, layers, in_channels, upsampling, weighting_mode):
        self._class_map = class_map
        heads = {}
        def create_single_head(in_channels, settings, upsampling):
            # The actual head
            task_head = MultiTaskHead(in_channels=in_channels,
                                      out_channels=settings['n_layers'],
                                      upsampling=upsampling)
            # The loss weighting for that head
            if settings['loss'] in ["CE", "BCE"] or weighting_mode == 'uniform':  # no weighting
                task_weighting = nn.Module()
                task_weighting.forward = lambda x: 0
            elif weighting_mode == 'layer_wise':  # a.k.a. homoscedatic
                task_weighting = LayerWeights(settings['n_layers'])
            elif weighting_mode == 'sample_wise':  # a.k.a. heteroscedastic
                task_weighting = MultiTaskHead(in_channels=in_channels,
                                               out_channels=settings['n_layers'],
                                               upsampling=upsampling)
            return WeightedHead([task_head, task_weighting])
        for name, settings in layers.items():
            if layers[name]['cls_specific_heads']:
                # Separate head for every class
                for cls_id in self._class_map.get_ids():
                    class_label = self._class_map.label_from_id(cls_id)
                    heads['{}_{}'.format(name, class_label)] = create_single_head(in_channels, settings, upsampling)
            else:
                heads[name] = create_single_head(in_channels, settings, upsampling)
        super(MultiTaskNet, self).__init__(heads)

    def forward(self, x):
        """Forward pass of input through module."""
        return {task_name: weighted_task_head(x) for task_name, weighted_task_head in self.items()}


class MultiTaskHead(nn.Sequential):
    """MultiTaskHead."""
    def __init__(self, in_channels, out_channels, upsampling=1):
        mid_channels = in_channels * 2
        modules = [nn.Conv2d(in_channels=in_channels,
                             out_channels=mid_channels,
                             kernel_size=1,
                             stride=1),
                   nn.BatchNorm2d(num_features=mid_channels),
                   nn.ReLU(inplace=True),
                   # nn.Dropout2d(p=0.5, inplace=False),
                   nn.Conv2d(in_channels=mid_channels,
                             out_channels=out_channels * upsampling ** 2,
                             kernel_size=1,
                             stride=1)]
        if upsampling != 1:
            modules.append(nn.PixelShuffle(upsampling))
        super(MultiTaskHead, self).__init__(*modules)


class WeightedHead(nn.ModuleList):
    def forward(self, x):
        return [module(x) for module in self]


class LayerWeights(nn.Module):
    def __init__(self, n_weights):
        super().__init__()
        self.register_parameter('weighting', nn.Parameter(Tensor(n_weights)))

    def forward(self, x):
        return self.weighting
