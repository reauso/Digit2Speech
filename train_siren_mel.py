import os

import cv2
import numpy as np
import ray
import torch
from cv2.mat_wrapper import Mat
from pytorch_msssim.ssim import _fspecial_gauss_1d, _ssim
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as functional
from torchvision.transforms import InterpolationMode

from data_handling.Dataset import DigitAudioDatasetForSpectrograms
from data_handling.util import map_numpy_values
from model.SirenModel import SirenModelWithFiLM


class CombinedLoss:
    def __init__(self):
        self.l1 = torch.nn.L1Loss()

        self.lambda_l1 = 1.0
        self.lambda_ms_ssim = 1.0

    def __call__(self, prediction, target):
        self.loss_l1 = self.l1(prediction, target)

        size = prediction.size()[1:3]
        size = [i * 4 for i in size]

        prediction_ms_ssim = functional.resize(prediction, size=size, interpolation=InterpolationMode.NEAREST)
        target_ms_ssim = functional.resize(target, size=size, interpolation=InterpolationMode.NEAREST)
        self.loss_ms_ssim = self.ms_ssim(X=prediction_ms_ssim, Y=target_ms_ssim, data_range=255)

        loss = self.loss_l1 * self.lambda_l1 + self.loss_ms_ssim * self.lambda_ms_ssim

        return loss

    def ms_ssim(
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


def parse_batch_size(encoded_batch_size, num_samples):
    if encoded_batch_size == 'per_file':
        return num_samples
    else:
        raise NotImplementedError("The encoded batch size '{}' is not implemented!".format(encoded_batch_size))


def train(config):
    torch.multiprocessing.set_start_method('spawn')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # create datasets and data loaders
    train_dataset = DigitAudioDatasetForSpectrograms(
        path=config['training_dataset_path'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=10, pin_memory=True,
                                      shuffle=config['shuffle_audio_files'], num_workers=4, drop_last=False)

    validation_dataset = DigitAudioDatasetForSpectrograms(
        path=config['validation_dataset_path'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=1, prefetch_factor=10, pin_memory=True,
                                           shuffle=False, num_workers=4, drop_last=False)

    # create model
    model = SirenModelWithFiLM(in_features=2,  # x coord and y coord
                               out_features=1,  # grayscale value of spectrogram at (x,y) coord
                               hidden_features=config["SIREN_hidden_features"],
                               num_layers=config["SIREN_num_layers"],
                               mod_features=config['num_mfccs'] * 4)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # load from checkpoint if checkpoint set
    '''if checkpoint_dir:
        load_file_path = os.path.join(checkpoint_dir, "checkpoint")
        print("Load from Checkpoint: {}".format(load_file_path))
        model_state, optimizer_state = torch.load(load_file_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)'''

    # necessary values and objects for training loop
    criterion = CombinedLoss()
    lambda_criterion = 1.0

    # training loop
    for epoch in range(config["epochs"]):
        train_losses = []
        eval_losses = []
        train_prediction_img = None
        eval_prediction_img = None

        for j, audio_file_data in enumerate(train_dataset_loader):
            # get batch data
            metadata, spectrogram, coordinates = audio_file_data
            spectrogram_shape = spectrogram.size()
            spectrogram = spectrogram.reshape((spectrogram_shape[0], spectrogram_shape[1], spectrogram_shape[2], 1))
            spectrogram = spectrogram.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device, non_blocking=True)

            # zero gradients
            optimizer.zero_grad()

            # get prediction
            prediction = model(coordinates, modulation_input)
            prediction_shape = spectrogram.size()
            prediction = prediction.reshape((prediction_shape[0], prediction_shape[1], prediction_shape[2], 1))
            # print('prediction: {}'.format(prediction))
            # print('prediction size: {}'.format(prediction.size()))

            # loss calculation
            loss = criterion(prediction, spectrogram) * lambda_criterion
            # print('loss: {}'.format(loss))

            # backpropagation
            loss.backward()
            optimizer.step()

            # documentation
            train_losses.append({
                'loss': loss.item(),
                'l1': criterion.loss_l1.item(),
                'ms_ssim': criterion.loss_ms_ssim.item(),
            })

            # image for tensorboard
            '''if j == len(train_dataset_loader) - 1:
                size = prediction.size()
                pred_img = prediction.permute(1, 0).view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                gt_img = spectrogram.permute(1, 0).view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                train_prediction_img = np.concatenate([pred_img, gt_img], axis=-1)'''

        for i, audio_file_data in enumerate(validation_dataset_loader):
            # get batch data
            metadata, spectrogram, coordinates = audio_file_data
            spectrogram_shape = spectrogram.size()
            spectrogram = spectrogram.reshape((spectrogram_shape[0], spectrogram_shape[1], spectrogram_shape[2], 1))
            spectrogram = spectrogram.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device, non_blocking=True)

            with torch.no_grad():
                prediction = model(coordinates, modulation_input)
                loss = criterion(prediction, spectrogram) * lambda_criterion
                eval_losses.append({
                    'loss': loss.item(),
                    'l1': criterion.loss_l1.item(),
                    'ms_ssim': criterion.loss_ms_ssim.item(),
                })

            # image for tensorboard
            '''if i == len(validation_dataset_loader) - 1:
                size = prediction.size()
                pred_img = prediction.permute(1, 0).view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                gt_img = spectrogram.permute(1, 0).view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                eval_prediction_img = np.concatenate([pred_img, gt_img], axis=-1)'''

        # save model after each epoch
        path = os.path.join(session.get_trial_dir(), "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

        # metrics
        metric_dict = {
            'train_loss': np.mean(np.array([losses['loss'] for losses in train_losses])),
            'train_l1': np.mean(np.array([losses['l1'] for losses in train_losses])),
            'train_ms_ssim': np.mean(np.array([losses['ms_ssim'] for losses in train_losses])),
            # 'train_vid': train_prediction_img,
            'eval_loss': np.mean(np.array([losses['loss'] for losses in eval_losses])),
            'eval_l1': np.mean(np.array([losses['l1'] for losses in eval_losses])),
            'eval_ms_ssim': np.mean(np.array([losses['ms_ssim'] for losses in eval_losses])),
            # 'eval_vid': eval_prediction_img,
        }
        tune.report(**metric_dict)


if __name__ == "__main__":
    # ray config
    num_trials = 1
    max_num_epochs = 30
    gpus_per_trial = 0.5

    # config
    config = {
        # data
        "training_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/training")),
        "validation_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/validation")),
        "num_mfccs": 50,  # tune.choice([20, 50, 128]),
        "feature_mapping_file": os.path.normpath(os.getcwd() + "/data_handling/feature_mapping.json"),
        'transformation_file': os.path.normpath(os.getcwd() + "/Dataset/transformation.json"),

        # data loading
        'batch_size': 'per_file',  # tune.choice([4096, 6144, 8192]),
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": 128,  # tune.choice([128, 256]),
        "SIREN_num_layers": 5,  # tune.choice([3, 5, 8]),
        "SIREN_mod_features": 256,  # tune.choice([128, 256, 348]),

        # training
        "lr": 0.001,  # tune.choice([0.001, 0.002, 0.003]),
        "epochs": 50,
    }

    env = {
        "working_dir": "./",
        "excludes": [".git"],
        "conda": "./environment.yml",
    }

    ray.init(address='auto', runtime_env=env, _node_ip_address="192.168.178.72")
    # ray.init()
    scheduler = ASHAScheduler(
        metric="eval_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["eval_loss", "training_iteration"])
    result = tune.run(
        train,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./Checkpoints",
        max_failures=1,  # Continues Trail from checkpoint if node is unavailable
        chdir_to_trial_dir=False,
    )

    best_trial = result.get_best_trial("eval_loss", "min", "last")
    print('Best trial Name: {}'.format(best_trial))
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["eval_loss"]))

    # train(config)
