import os

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import DataLoader

from data_handling.Dataset import DigitAudioDataset
from model.SirenModel import SirenModelWithFiLM


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
    train_dataset = DigitAudioDataset(
        path=config['training_dataset_path'],
        audio_sample_coverage=config['audio_sample_coverage'],
        shuffle_audio_samples=config['shuffle_audio_samples'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
        transformation_file=config['transformation_file'],
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=1, prefetch_factor=10, pin_memory=True,
                                      shuffle=config['shuffle_audio_files'], num_workers=4, drop_last=False)

    validation_dataset = DigitAudioDataset(
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
    model = SirenModelWithFiLM(in_features=2,
                               out_features=3,
                               hidden_features=config["SIREN_hidden_features"],
                               num_layers=config["SIREN_num_layers"],
                               mod_features=config['num_mfccs'] + 3)
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
    criterion = torch.nn.MSELoss(reduction='mean')
    #criterion = get_log_cosh_loss
    lambda_criterion = 1.0

    # training loop
    for epoch in range(config["epochs"]):
        train_losses = []
        eval_losses = []

        for j, audio_file_data in enumerate(train_dataset_loader):
            # get batch data
            metadata, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0).to(device, non_blocking=True)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0).to(device, non_blocking=True)

            # print status
            # print("Audio File {}/{} with {} samples".format(j, len(train_dataset_loader), audio_samples.size()[0]))

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device, non_blocking=True)

            num_samples = audio_samples.size()[0]
            batch_size = parse_batch_size(config['batch_size'], num_samples) if isinstance(config['batch_size'], str) else config['batch_size']
            num_batches = int(num_samples / batch_size)
            num_batches = num_batches if num_samples % batch_size == 0 else num_batches + 1
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, num_samples)

                audio_sample_batch = audio_samples[start_index: end_index]
                audio_sample_indices_batch = audio_sample_indices[start_index: end_index]

                # zero gradients
                optimizer.zero_grad()

                # get prediction
                prediction = model(audio_sample_indices_batch, modulation_input)
                #print('prediction: {}'.format(prediction))

                # loss calculation
                loss = criterion(prediction, audio_sample_batch) * lambda_criterion
                #print('loss: {}'.format(loss))

                # backpropagation
                loss.backward()
                optimizer.step()

                # documentation
                train_losses.append(loss.item())

        for i, audio_file_data in enumerate(validation_dataset_loader):
            # get batch data
            metadata, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0).to(device, non_blocking=True)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0).to(device, non_blocking=True)

            # print status
            # print("Audio File {}/{} with {} samples".format(i, len(train_dataset_loader), audio_samples.size()[0]))

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device, non_blocking=True)

            with torch.no_grad():
                prediction = model(audio_sample_indices, modulation_input)
                loss = criterion(prediction, audio_samples) * lambda_criterion
                eval_losses.append(loss.item())

        # save model after each epoch
        path = os.path.join(session.get_trial_dir(), "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

        # metrics
        metric_dict = {
            'train_loss': np.mean(np.array(train_losses)),
            'eval_loss': np.mean(np.array(eval_losses)),
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
        "audio_sample_coverage": 1.0,  # tune.choice([0.3, 0.6, 0.9]),
        "shuffle_audio_samples": False,  # tune.choice([True, False]),
        "num_mfccs": 50,  # tune.choice([20, 50, 128]),
        "feature_mapping_file": os.path.normpath(os.getcwd() + "/data_handling/feature_mapping.json"),
        'transformation_file': os.path.normpath(os.getcwd() + "/Dataset/transformation.json"),

        # data loading
        'batch_size': 'per_file',  # tune.choice([4096, 6144, 8192]),
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": tune.choice([128, 256]),
        "SIREN_num_layers": tune.choice([3, 5, 8]),
        "SIREN_mod_features": tune.choice([128, 256, 348]),

        # training
        "lr": tune.choice([0.001, 0.002, 0.003]),
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
