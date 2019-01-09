"""Neural network model."""

from torch import nn

class Model(nn.Module):
    """Neural network module."""

    def __init__(self, configs):
        super().__init__()
        self._configs = configs
        self._encoder = self._create_encoder()
        self._decoder = self._create_decoder()

    def _create_encoder(self):
        # TODO:
        pass

    def _create_decoder(self):
        # TODO:
        pass

    def forward(self, input_data):
        return self._encoder(self._decoder(input_data))
