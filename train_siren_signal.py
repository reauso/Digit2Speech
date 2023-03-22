import os

import cv2
import numpy as np
import ray
import torch
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_handling.Dataset import DigitAudioDatasetForSignal
from util.array_helper import signal_to_image
from model.SirenModel import MappingType, SirenModelWithFiLM


def get_log_cosh_loss(prediction, target):
    loss = torch.mean(torch.log(torch.cosh(prediction - target)))
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
    train_dataset = DigitAudioDatasetForSignal(
        path=config['training_dataset_path'],
        audio_sample_coverage=config['audio_sample_coverage'],
        shuffle_audio_samples=config['shuffle_audio_samples'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
        transformation_file=config['transformation_file'],
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=10, pin_memory=True,
                                      shuffle=config['shuffle_audio_files'], num_workers=4, drop_last=False)

    validation_dataset = DigitAudioDatasetForSignal(
        path=config['validation_dataset_path'],
        audio_sample_coverage=1.0,
        shuffle_audio_samples=False,
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
        transformation_file=config['transformation_file'],
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=1, prefetch_factor=10, pin_memory=True,
                                           shuffle=False, num_workers=4, drop_last=False)

    # create model
    model = SirenModelWithFiLM(
        in_features=1,
        out_features=1,
        hidden_features=config['SIREN_hidden_features'],
        hidden_layers=config['SIREN_hidden_layers'],
        mod_in_features=config['num_mfccs'] * 4,
        mod_features=config['MODULATION_hidden_features'],
        mod_hidden_layers=config['MODULATION_hidden_layers'],
        modulation_type=config['MODULATION_Type'],
    )
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # necessary values and objects for training loop
    criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = get_log_cosh_loss
    lambda_criterion = 1.0

    # epoch loop
    for epoch in range(config["epochs"]):
        train_losses = []
        eval_losses = []
        eval_prediction_img = None
        train_prediction_img = None

        # training loop
        for i, audio_file_data in enumerate(train_dataset_loader):
            # get batch data
            metadata, _, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0)

            # tensors to device
            audio_samples = audio_samples.to(device, non_blocking=True)
            audio_sample_indices = audio_sample_indices.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            # get prediction
            prediction = model(audio_sample_indices, metadata)

            # loss calculation
            loss = criterion(prediction, audio_samples)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # documentation
            train_losses.append({'loss': loss.item()})

            # image for tensorboard
            if i == len(validation_dataset_loader) - 1:
                train_prediction_img = image_for_tensorboard(prediction, audio_samples)

        # validation loop
        for i, audio_file_data in enumerate(validation_dataset_loader):
            # get batch data
            metadata, _, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0)

            # tensors to device
            audio_samples = audio_samples.to(device, non_blocking=True)
            audio_sample_indices = audio_sample_indices.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            with torch.no_grad():
                prediction = model(audio_sample_indices, metadata)
                loss = criterion(prediction, audio_samples) * lambda_criterion
                eval_losses.append({'loss': loss.item()})

            # image for tensorboard
            if i == len(validation_dataset_loader) - 1:
                eval_prediction_img = image_for_tensorboard(prediction, audio_samples)

        # save model after each epoch
        path = os.path.join(session.get_trial_dir(), "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

        # metrics
        train_losses = {'train_{}'.format(key): np.mean(np.array([losses[key] for losses in train_losses]))
                        for key in train_losses[0].keys()}
        eval_losses = {'eval_{}'.format(key): np.mean(np.array([losses[key] for losses in eval_losses]))
                       for key in eval_losses[0].keys()}
        metric_dict = {
            'train_vid': train_prediction_img,
            'eval_vid': eval_prediction_img,
        }
        metric_dict.update(train_losses)
        metric_dict.update(eval_losses)
        tune.report(**metric_dict)


def image_for_tensorboard(prediction, ground_truth):
    prediction = prediction.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    prediction = prediction.reshape(prediction.shape[0])
    ground_truth = ground_truth.reshape(ground_truth.shape[0])

    pred_img = signal_to_image(prediction)[:, :, :3]
    gt_img = signal_to_image(ground_truth)[:, :, :3]

    eval_prediction_img = np.concatenate([pred_img, gt_img], axis=1)
    scale_factor = 800 / eval_prediction_img.shape[1]
    eval_prediction_img = cv2.resize(eval_prediction_img, (0, 0), fx=scale_factor, fy=scale_factor)
    eval_prediction_img = eval_prediction_img.transpose(2, 0, 1)
    eval_prediction_img = eval_prediction_img.reshape(1, 1, *eval_prediction_img.shape)

    return eval_prediction_img


if __name__ == "__main__":
    # ray config
    num_trials = 40
    max_num_epochs = 40
    gpus_per_trial = 1

    # config
    config = {
        # data
        "training_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/training")),
        "validation_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/validation")),
        "audio_sample_coverage": 1.0,  # tune.choice([0.3, 0.6, 0.9]),
        "shuffle_audio_samples": False,  # tune.choice([True, False]),
        "num_mfccs": 50,  # tune.choice([20, 50, 128]),
        "feature_mapping_file": os.path.normpath(os.getcwd() + "/data_handling/feature_mapping.json"),
        'transformation_file': os.path.normpath(os.getcwd() + "/Dataset/transformation.json"),

        # data loading
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": tune.choice([128, 256, 384, 512]),
        "SIREN_hidden_layers": tune.choice([3, 5, 8]),
        "MODULATION_Type": tune.choice(list(MappingType)),
        "MODULATION_hidden_features": tune.choice([128, 256, 384, 512]),
        "MODULATION_hidden_layers": tune.choice([3, 5, 8]),
        
        # training
        "lr": tune.choice([0.00005, 0.000075, 0.0001]),
        "epochs": 50,
    }

    env = {
        "working_dir": "./",
        "excludes": [".git"],
        "conda": "./environment.yml",
    }

    '''def ray_mapping(config_entry):
        if type(config_entry) is ray.tune.search.sample.Categorical:
            entry = config_entry.sample()
        else:
            entry = config_entry
        return entry
    config = {key: ray_mapping(value) for key, value in config.items()}
    train(config)'''

    # ray.init(address='auto', runtime_env=env, _node_ip_address="192.168.178.72")
    ray.init()
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
