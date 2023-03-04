import os

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import DataLoader
from pytorch_msssim import ms_ssim

from data_handling.Dataset import DigitAudioDatasetForSpectrograms
from model.SirenModel import SirenModelWithFiLM


class CombinedLoss:
    def __init__(self):
        self.l1 = torch.nn.L1Loss()
        self.ms_ssim = ms_ssim

        self.lambda_l1 = 1.0
        self.lambda_ms_ssim = 1.0

    def __call__(self, prediction, target):
        self.loss_l1 = self.l1(prediction, target)

        prediction_ms_ssim = (prediction + 1) * 0.5
        target_ms_ssim = (target + 1) * 0.5

        print(prediction_ms_ssim.shape)
        print(target_ms_ssim.shape)
        self.loss_ms_ssim = self.ms_ssim(prediction_ms_ssim, target_ms_ssim, data_range=1)

        loss = self.loss_l1 * self.lambda_l1 + self.loss_ms_ssim * self.lambda_ms_ssim
        return loss


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

        # save model after each epoch
        path = os.path.join(session.get_trial_dir(), "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

        # metrics
        metric_dict = {
            'train_loss': np.mean(np.array([losses['loss'] for losses in train_losses])),
            'train_l1': np.mean(np.array([losses['l1'] for losses in train_losses])),
            'train_ms_ssim': np.mean(np.array([losses['ms_ssim'] for losses in train_losses])),
            'eval_loss': np.mean(np.array([losses['loss'] for losses in eval_losses])),
            'eval_l1': np.mean(np.array([losses['l1'] for losses in eval_losses])),
            'eval_ms_ssim': np.mean(np.array([losses['ms_ssim'] for losses in eval_losses])),
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

    '''ray.init(address='auto', runtime_env=env, _node_ip_address="192.168.178.72")
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
        best_trial.last_result["eval_loss"]))'''

    train(config)
