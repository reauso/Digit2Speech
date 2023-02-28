import torch
import torch.nn as nn
import numpy as np
from model.ModulationLayer import FiLM, MappingNetwork
from model.initialization import init_network_parameters


class SineLayerWithFilm(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, use_film=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.use_film = use_film

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.film = FiLM()

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x, scale=None, shift=None):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(x)
        if self.use_film:
            intermediate = self.film(intermediate, scale, shift)
        return torch.sin(intermediate)


class SirenModelWithFiLM(nn.Module):

    def __init__(self, in_features, out_features, hidden_features, num_layers, mod_features, mod_layer=3, use_film=True, omega_0=30):
        super(SirenModelWithFiLM, self).__init__()

        # First Layer
        self.siren_net = nn.ModuleList([SineLayerWithFilm(in_features, hidden_features, is_first=True, use_film=use_film, omega_0=omega_0)])
        # Hidden Layer
        self.siren_net.extend([SineLayerWithFilm(hidden_features, hidden_features, use_film=use_film, omega_0=omega_0)] * (num_layers - 2))
        # Output Layer
        self.siren_net.append(SineLayerWithFilm(hidden_features, out_features, use_film=False, omega_0=omega_0))

        self.modulation_net = MappingNetwork(mod_features, hidden_features, mod_layer)

        init_network_parameters(self)

    def forward(self, x, modulation_input):

        scale, shift = self.modulation_net(modulation_input)

        for layer in self.siren_net:
            x = layer(x, scale, shift)

        return x


class SirenModel(nn.Module):

    def __init__(self, in_features, hidden_features, num_layers, out_features, omega_0=30):
        super(SirenModel, self).__init__()

        # First Layer
        self.siren_net = nn.ModuleList(
            [SineLayerWithFilm(in_features, hidden_features, is_first=True, use_film=False, omega_0=omega_0)])
        # Hidden Layer*
        self.siren_net.extend(
            [SineLayerWithFilm(hidden_features, hidden_features, use_film=False, omega_0=omega_0)] * (num_layers - 2))
        # Output Layer
        self.siren_net.append(SineLayerWithFilm(hidden_features, out_features, use_film=False, omega_0=omega_0))

    def forward(self, x,):

        for layer in self.siren_net:
            x = layer(x)

        return x


if __name__ == "__main__":

    siren = SirenModelWithFiLM(2, 256, 4, 3, 20)

    # Test Input
    batch_size = 8
    sample_rel_cord = torch.rand((8, 2))
    sample_exp = torch.rand((8, 20))

    siren(sample_rel_cord, sample_exp)
