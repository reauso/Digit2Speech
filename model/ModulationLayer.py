import torch
import torch.nn as nn
import unittest

from model.PositionalEncoding import HarmonicEmbedding


class MappingNetwork(nn.Module):

    def __init__(self, input_size, output_size, num_dimensions, num_features=256, hidden_layers=3,
                 use_harmonic_embedding=True, num_harmonic_functions=4):
        super(MappingNetwork, self).__init__()
        self.num_dimensions = num_dimensions
        self.output_size = output_size
        self.use_harmonic_embedding = use_harmonic_embedding

        # harmonic embedding
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions=num_harmonic_functions)

        # modulation layers
        input_size = input_size * num_harmonic_functions * 2 if self.use_harmonic_embedding else input_size
        self.modulation_layers = torch.nn.ModuleList([nn.Linear(input_size, num_features)])
        for i in range(hidden_layers - 1):
            self.modulation_layers.extend([nn.Linear(num_features, num_features)])

        # final scale and shift layers
        self.lin_scale = nn.Linear(num_features, self.num_dimensions * self.output_size)
        self.lin_shift = nn.Linear(num_features, self.num_dimensions * self.output_size)

        # activation function
        self.relu = nn.LeakyReLU(0.2,)

    def forward(self, x):
        if self.use_harmonic_embedding:
            x = self.harmonic_embedding(x)

        for layer in self.modulation_layers:
            x = self.relu(layer(x))

        scale = self.lin_scale(x).reshape((self.num_dimensions, x.size()[0], self.output_size))
        shift = self.lin_shift(x).reshape((self.num_dimensions, x.size()[0], self.output_size))

        return scale, shift


class FiLM(nn.Module):

    def __init__(self,):
        super(FiLM, self).__init__()

    def forward(self, x, scale, shift):
        assert scale.size() == shift.size(), "Scale and Shift have different dimensions in Film Layer"
        assert scale.size()[-1] == x.size()[-1], "Different number of features"

        return scale * x + shift


class FiLMTest(unittest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.num_features = 3
        self.modulations_dim = 53

        self.film_layer = FiLM()
        self.film_layer2 = FiLM()
        self.mapping_network = MappingNetwork(self.modulations_dim, self.num_features)

    def test_1d(self):
        x = torch.ones((self.batch_size, self.num_features))  # Batchsize 8 and 32 Features
        scale = torch.ones((self.batch_size, self.num_features)) * 2  # Batchsize 8 and 32 Features
        shift = torch.ones((self.batch_size, self.num_features)) * 3  # Batchsize 8 and 32 Features
        x_out = x * 2 + 3

        self.assertTrue(torch.equal(self.film_layer(x, scale, shift), x_out), "FiLM-1D failed")

    def test_2d(self):
        x = torch.rand((self.batch_size, 5, self.num_features))  # Batchsize 8 and 32 Features
        scale = torch.ones((self.batch_size, self.num_features)) * 2  # Batchsize 8 and 32 Features
        shift = torch.ones((self.batch_size, self.num_features)) * 3  # Batchsize 8 and 32 Features
        print(x.size())
        print(scale.size())
        print(shift.size())
        x_out = x * 2 + 3

        self.assertTrue(torch.equal(self.film_layer(x, scale, shift), x_out), "FiLM-2D failed")

    def test_two_1D_FiLMs_with_MappingNetwork(self):
        """
        Example for Using
        =================
        Depending on def setUp
        """

        modulation = torch.rand((self.batch_size, self.modulations_dim))
        scale, shift = self.mapping_network(modulation)

        x = torch.ones((self.batch_size, self.num_features))
        x = self.film_layer(x, scale, shift)
        x = self.film_layer2(x, scale, shift)

        print("Finish Example!")
        return x


if __name__ == "__main__":
    #unittest.main()
    net = FiLM()
    input1 = torch.zeros((8, 128))
    input = torch.zeros((8, 128))
    net(input1, input, input)
