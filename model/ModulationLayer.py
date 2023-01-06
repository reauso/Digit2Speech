"""
Modulation Layer for PyTorch
Author: Alexander Pech
Last-Edit: 20.05.22

- Feature-wise Linear Modulation (FiLM)
"""
import torch
import torch.nn as nn
import unittest


class MappingNetwork(nn.Module):

    def __init__(self, modulation_input_size, num_features, modulation_layer=3):
        super(MappingNetwork, self).__init__()

        #self.lin0 = nn.Linear(modulation_input_size, 256)
        #self.lin1 = nn.Linear(256, 256)
        #self.lin2 = nn.Linear(256, 256)

        self.modulation_layers = torch.nn.ModuleList([nn.Linear(modulation_input_size, 256)])
        for i in range(modulation_layer-1):
            self.modulation_layers.extend([nn.Linear(256, 256)])

        self.lin_scale = nn.Linear(256, num_features)
        self.lin_shift = nn.Linear(256, num_features)

        self.relu = nn.LeakyReLU(0.2,)

    def forward(self, x):

        for layer in self.modulation_layers:
            x = self.relu(layer(x))
        # x = self.relu(self.lin1(x))
        # x = self.relu(self.lin2(x))

        scale = self.lin_scale(x)
        shift = self.lin_shift(x)

        return scale, shift


class FiLM(nn.Module):

    def __init__(self,):
        super(FiLM, self).__init__()

    def forward(self, x, scale, shift):

        assert scale.size() == shift.size(), "Scale and Shift have different dimensions in Film Layer"
        assert scale.size()[1] == x.size()[1], "Different number of features"

        print(x.size())
        #
        if len(x.size())-1 == 2:  # batch + 2d
            print('A')
            scale = scale.unsqueeze(-1).expand_as(x)
            shift = shift.unsqueeze(-1).expand_as(x)
            print(scale.size())
        elif len(x.size())-1 == 3:  # batch + 2d
            print('B')
            scale = scale.unsqueeze(-1).unsqueeze(-1).expand_as(x)
            shift = shift.unsqueeze(-1).unsqueeze(-1).expand_as(x)
            print(scale.size())

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
        x = torch.rand((self.batch_size, self.num_features, self.num_features))  # Batchsize 8 and 32 Features
        scale = torch.ones((self.batch_size, self.num_features)) * 2  # Batchsize 8 and 32 Features
        shift = torch.ones((self.batch_size, self.num_features)) * 3  # Batchsize 8 and 32 Features
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
