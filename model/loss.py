import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as functional
from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
from torchvision.transforms import InterpolationMode


class CombinedLoss:
    def __init__(self, device=None):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.l1 = torch.nn.L1Loss()

        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_sobel = GradLoss().to(device)

        self.losses = {}

        self.lambdas = {
            'l1': 0.0,
            'mse': 0.75,
            'sobel': 0.2,
            'ms_ssim': 0.4,
        }
        self.loss_names = list(self.lambdas.keys())

    def __call__(self, prediction, target):
        self.calculate_individual_losses(prediction, target)

        # sum
        all_losses_list = [self.losses[name] * self.lambdas[name] for name in self.loss_names]
        loss = torch.sum(torch.stack(all_losses_list))

        return loss, self.losses

    def calculate_individual_losses(self, prediction, target):
        self.losses['l1'] = self.l1(prediction, target)
        self.losses['mse'] = self.criterion_mse(prediction, target)

        prediction_sobel = prediction.permute(0, 3, 1, 2)
        target_sobel = target.permute(0, 3, 1, 2)
        self.losses['sobel'] = self.criterion_sobel(prediction_sobel, target_sobel)

        size = prediction.size()[1:3]
        size = [i * 4 for i in size]
        prediction_ms_ssim = functional.resize(prediction, size=size, interpolation=InterpolationMode.NEAREST)
        target_ms_ssim = functional.resize(target, size=size, interpolation=InterpolationMode.NEAREST)
        self.losses['ms_ssim'] = 1 - self.__ms_ssim(X=prediction_ms_ssim, Y=target_ms_ssim, data_range=255)

    def __ms_ssim(
            self, X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None,
            K=(0.01, 0.03)
    ):

        r""" interface of ms-ssim
        Args:
            X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        Returns:
            torch.Tensor: ms-ssim results
        """
        if not X.shape == Y.shape:
            raise ValueError("Input images should have the same dimensions.")

        '''for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)'''

        if not X.type() == Y.type():
            raise ValueError("Input images should have the same dtype.")

        if len(X.shape) == 4:
            avg_pool = F.avg_pool2d
        elif len(X.shape) == 5:
            avg_pool = F.avg_pool3d
        else:
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if win is not None:  # set win_size
            win_size = win.shape[-1]

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        smaller_side = min(X.shape[-2:])
        assert smaller_side > (win_size - 1) * (
                2 ** 4
        ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.FloatTensor(weights).to(X.device, dtype=X.dtype)

        if win is None:
            win = _fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

        levels = weights.shape[0]
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

            if i < levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s % 2 for s in X.shape[2:]]
                X = avg_pool(X, kernel_size=2, padding=padding)
                Y = avg_pool(Y, kernel_size=2, padding=padding)

        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

        if size_average:
            return ms_ssim_val.mean()
        else:
            return ms_ssim_val.mean(1)


# from https://github.com/zhaoyuzhi/PyTorch-Sobel/blob/main/pytorch-sobel.py
class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        '''
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)
