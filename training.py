import os
import numpy as np
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from torch.utils.data import DataLoader

from data_handling.Dataset import DigitAudioDataset
from model.SirenModel import SirenModelWithFiLM


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
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=1,
                                      shuffle=config['shuffle_audio_files'], num_workers=0, drop_last=False)

    validation_dataset = DigitAudioDataset(
        path=config['validation_dataset_path'],
        audio_sample_coverage=1.0,
        shuffle_audio_samples=False,
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=1,
                                           shuffle=False, num_workers=0, drop_last=False)

    # create model
    model = SirenModelWithFiLM(in_features=1,
                               out_features=1,
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
    criterion = nn.MSELoss()

    # training loop
    for epoch in range(config["epochs"]):
        train_losses = []
        eval_losses = []

        for audio_file_data in train_dataset_loader:
            # get batch data
            metadata, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0).to(device)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0).to(device)

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device)

            num_samples = audio_samples.size()[0]
            num_batches = int(num_samples / config['batch_size'])
            num_batches = num_batches if num_samples % config['batch_size'] == 0 else num_batches + 1
            for i in range(num_batches):
                start_index = i * config['batch_size']
                end_index = min((i + 1) * config['batch_size'], num_samples)

                audio_sample_batch = audio_samples[start_index: end_index]
                audio_sample_indices_batch = audio_sample_indices[start_index: end_index]

                # zero gradients
                optimizer.zero_grad()

                # get prediction
                prediction = model(audio_sample_indices_batch, modulation_input)

                # loss calculation
                loss = criterion(prediction, audio_sample_batch)

                # backpropagation
                loss.backward()
                optimizer.step()

                # documentation
                train_losses.append(loss.item())

        for audio_file_data in validation_dataset_loader:
            # get batch data
            metadata, audio_samples, audio_sample_indices = audio_file_data
            audio_samples = torch.transpose(audio_samples, 1, 0).to(device)
            audio_sample_indices = torch.transpose(audio_sample_indices, 1, 0).to(device)

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1).to(device)

            with torch.no_grad():
                prediction = model(audio_sample_indices, modulation_input)
                loss = criterion(prediction, audio_samples)
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
    num_samples = 1
    max_num_epochs = 30
    gpus_per_trial = 1

    # config
    config = {
        # data
        "training_dataset_path": os.path.normpath("Dataset/training"),
        "validation_dataset_path": os.path.normpath("Dataset/validation"),
        "audio_sample_coverage": tune.quniform(0.2, 0.5, 0.1),
        "shuffle_audio_samples": tune.choice([True, False]),
        "num_mfccs": 50,
        "feature_mapping_file": os.path.normpath("data_handling/feature_mapping.json"),

        # data loading
        'batch_size': tune.choice([8, 64, 128, 512, 1024]),
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": tune.choice([128]),
        "SIREN_num_layers": tune.choice([5]),
        "SIREN_mod_features": tune.choice([348]),

        # training
        "lr": 0.001,
        "epochs": 50,
    }

    env = {
        "working_dir": "./",
    }

    #ray.init(address='auto', runtime_env=env,)
    ray.init()
    scheduler = ASHAScheduler(
        metric="eval_loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["eval_loss", "training_iteration"])
    result = tune.run(
        train,
        resources_per_trial={"cpu": 10, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="./Checkpoints",
        max_failures=1,  # Continues Trail from checkpoint if node is unavailable
        chdir_to_trial_dir=False,
    )

    best_trial = result.get_best_trial("eval_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["eval_loss"]))

    #train(config)
