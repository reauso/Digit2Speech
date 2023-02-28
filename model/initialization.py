from torch.nn import init


def init_network_parameters(network, init_gain=0.02):

    def init_function(m):
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif class_name.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    network.apply(init_function)
