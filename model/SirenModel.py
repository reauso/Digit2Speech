import torch
import torch.nn as nn
import numpy as np
from model.ModulationLayer import FiLM, MappingNetwork


class SineLayerWithFilm(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.film = FiLM()

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input, scale, shift):
        intermediate = self.omega_0 * self.linear(input)
        intermediate = self.film(intermediate, scale, shift)
        return torch.sin(intermediate)


class SirenModelWithFiLM(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers,
                 mod_in_features, mod_features=256, mod_hidden_layers=3, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.sine_layers = []
        self.final_layers = []
        self.sine_layers.append(SineLayerWithFilm(in_features, hidden_features,
                                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.sine_layers.append(SineLayerWithFilm(hidden_features, hidden_features,
                                                      is_first=False, omega_0=hidden_omega_0))

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                         np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.final_layers.append(final_linear)
        #self.final_layers.append(torch.nn.Tanh())

        self.sine_layers = nn.Sequential(*self.sine_layers)
        self.final_layers = nn.Sequential(*self.final_layers)
        self.modulation_network = MappingNetwork(
            input_size=mod_in_features,
            output_size=hidden_features,
            num_dimensions=hidden_layers,
            num_features=mod_features,
            hidden_layers=mod_hidden_layers,
        )

    def forward(self, x, modulation_input):
        scale, shift = self.modulation_network(modulation_input)

        for i, layer in enumerate(self.sine_layers):
            x = layer(x, scale[:, i], shift[:, i])

        x = self.final_layers(x)

        return torch.sin(x)
