from enum import Enum

import torch
import torch.nn as nn
import numpy as np
from model.ModulationLayer import FiLM, MappingNetwork
from model.PositionalEncoding import HarmonicEmbedding


class MappingType(Enum):
    One_Network_One_Dimension_For_All_Layers = 'One_Network_One_Dimension_For_All_Layers'
    One_Network_Mult_Dimension_For_Each_Layer = 'One_Network_Mult_Dimension_For_Each_Layer'
    Mult_Networks_One_Dimension_For_Each_Layer = 'Mult_Networks_One_Dimension_For_Each_Layer'

    def __str__(self):
        return self.value


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
    def __init__(
            self,
            in_features,
            out_features,
            hidden_features,
            hidden_layers,
            mod_in_features,
            mod_features=256,
            mod_hidden_layers=3,
            first_omega_0=30,
            hidden_omega_0=30.,
            modulation_type: MappingType = MappingType.Mult_Networks_One_Dimension_For_Each_Layer,
            use_harmonic_embedding=True,
            num_harmonic_embeddings=30,
            use_mod_harmonic_embedding=True,
            num_mod_harmonic_embedding=4,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.mapping_type = modulation_type
        self.use_harmonic_embedding = use_harmonic_embedding

        self.sine_layers = []
        self.final_layers = []

        # harmonic embedding
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions=num_harmonic_embeddings)

        # sine layers
        in_features = in_features * num_harmonic_embeddings * 2 if self.use_harmonic_embedding else in_features
        self.sine_layers.append(SineLayerWithFilm(in_features, hidden_features,
                                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.sine_layers.append(SineLayerWithFilm(hidden_features, hidden_features,
                                                      is_first=False, omega_0=hidden_omega_0))

        # final layers
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                         np.sqrt(6 / hidden_features) / hidden_omega_0)

        self.final_layers.append(final_linear)

        # as sequential
        self.sine_layers = nn.Sequential(*self.sine_layers)
        self.final_layers = nn.Sequential(*self.final_layers)

        # modulation network(s)
        self.modulation_networks = self._get_modulation(mod_in_features, hidden_features,
                                                        hidden_layers, mod_features, mod_hidden_layers,
                                                        use_mod_harmonic_embedding, num_mod_harmonic_embedding)
        self.modulation_dimension_mapping = {
            MappingType.One_Network_One_Dimension_For_All_Layers: 1,
            MappingType.One_Network_Mult_Dimension_For_Each_Layer: hidden_layers + 1,
            MappingType.Mult_Networks_One_Dimension_For_Each_Layer: hidden_layers + 1,
        }

    def _get_modulation(
            self,
            mod_in_features,
            hidden_features,
            hidden_layers,
            mod_features,
            mod_hidden_layers,
            use_mod_harmonic_embedding,
            num_mod_harmonic_embedding,
    ):
        if self.mapping_type is MappingType.One_Network_One_Dimension_For_All_Layers:
            modulation_network = torch.nn.ModuleList([
                MappingNetwork(
                    input_size=mod_in_features,
                    output_size=hidden_features,
                    num_dimensions=1,
                    num_features=mod_features,
                    hidden_layers=mod_hidden_layers,
                    use_harmonic_embedding=use_mod_harmonic_embedding,
                    num_harmonic_functions=num_mod_harmonic_embedding,
                )])
        elif self.mapping_type is MappingType.One_Network_Mult_Dimension_For_Each_Layer:
            modulation_network = torch.nn.ModuleList([
                MappingNetwork(
                    input_size=mod_in_features,
                    output_size=hidden_features,
                    num_dimensions=hidden_layers + 1,
                    num_features=mod_features,
                    hidden_layers=mod_hidden_layers,
                    use_harmonic_embedding=use_mod_harmonic_embedding,
                    num_harmonic_functions=num_mod_harmonic_embedding,
                )])
        elif self.mapping_type is MappingType.Mult_Networks_One_Dimension_For_Each_Layer:
            modulation_network = torch.nn.ModuleList([
                MappingNetwork(
                    input_size=mod_in_features,
                    output_size=hidden_features,
                    num_dimensions=1,
                    num_features=mod_features,
                    hidden_layers=mod_hidden_layers,
                    use_harmonic_embedding=use_mod_harmonic_embedding,
                    num_harmonic_functions=num_mod_harmonic_embedding,
                ) for _ in range(hidden_layers + 1)
            ])
        else:
            raise NotImplementedError('MappingType not implemented {}'.format(self.mapping_type))

        return modulation_network

    def forward(self, x, modulation_input):
        # harmonic embedding
        if self.use_harmonic_embedding:
            x = self.harmonic_embedding(x)

        # modulation
        scale, shift = self.calculate_scale_and_shift(modulation_input)

        # sine layers
        mod_dimensions = self.modulation_dimension_mapping[self.mapping_type]
        for i, layer in enumerate(self.sine_layers):
            mod_index = min(i, mod_dimensions - 1)
            x = layer(x, scale[mod_index], shift[mod_index])

        # final layers
        x = self.final_layers(x)

        return torch.sin(x)

    def calculate_scale_and_shift(self, modulation_input):
        dimension = self.modulation_dimension_mapping[self.mapping_type]
        mod_batch_size = modulation_input.size()[0]
        device = modulation_input.device

        scale = torch.zeros((dimension, mod_batch_size, self.hidden_features), device=device)
        shift = torch.zeros((dimension, mod_batch_size, self.hidden_features), device=device)

        for i, modulation_network in enumerate(self.modulation_networks):
            current_scale, current_shift = modulation_network(modulation_input)
            scale[i] = current_scale[0]
            shift[i] = current_shift[0]
        return scale, shift
