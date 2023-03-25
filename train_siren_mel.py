import os
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from data_handling.Dataset import DigitAudioDatasetForSpectrograms
from ray_result_analyzer import RayTuneAnalysis, TextTableView
from util.array_helper import map_numpy_values
from model.SirenModel import SirenModelWithFiLM, MappingType
from model.loss import CombinedLoss
from util.checkpoint_helper import nearest_experiment_path, best_trial_path
from util.data_helper import write_textfile


def train(config):
    torch.multiprocessing.set_start_method('spawn')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # create datasets and data loaders
    train_dataset = DigitAudioDatasetForSpectrograms(
        path=config['training_dataset_path'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True, prefetch_factor=10,
                                      shuffle=config['shuffle_audio_files'], num_workers=4, drop_last=False)

    validation_dataset = DigitAudioDatasetForSpectrograms(
        path=config['validation_dataset_path'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=1, pin_memory=True, prefetch_factor=10,
                                           shuffle=False, num_workers=4, drop_last=False)

    # create model
    model = SirenModelWithFiLM(
        in_features=2,  # x coord and y coord
        out_features=1,  # grayscale value of spectrogram at (x,y) coord
        hidden_features=config['SIREN_hidden_features'],
        hidden_layers=config['SIREN_hidden_layers'],
        mod_in_features=config['num_mfccs'] * 4,
        mod_features=config['MODULATION_hidden_features'],
        mod_hidden_layers=config['MODULATION_hidden_layers'],
        modulation_type=config['MODULATION_Type'],
        use_harmonic_embedding=config['SIREN_use_harmonic_embedding'],
        num_harmonic_functions=config['SIREN_num_harmonic_functions'],
        use_mod_harmonic_embedding=config['MODULATION_use_harmonic_embedding'],
        num_mod_harmonic_functions=config['MODULATION_num_harmonic_functions'],
    )
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])

    # necessary values and objects for training loop
    criterion = CombinedLoss(device)

    # epoch loop
    for epoch in range(config["epochs"]):
        train_losses = []
        eval_losses = []
        train_prediction_img = None
        eval_prediction_img = None

        # training loop
        for i, data_pair in enumerate(train_dataset_loader):
            # get batch data
            metadata, _, spectrogram, coordinates = data_pair
            spectrogram_shape = spectrogram.size()
            spectrogram = spectrogram.reshape((spectrogram_shape[0], spectrogram_shape[1], spectrogram_shape[2], 1))

            # tensors to device
            spectrogram = spectrogram.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            # get prediction
            prediction = model(coordinates, metadata)
            prediction = prediction.reshape((spectrogram_shape[0], spectrogram_shape[1], spectrogram_shape[2], 1))

            # loss calculation
            loss, individual_losses = criterion(prediction, spectrogram)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # documentation
            individual_losses['loss'] = loss
            individual_losses = {key: value.item() for key, value in individual_losses.items()}
            train_losses.append(individual_losses)

            # image for tensorboard
            if i == len(train_dataset_loader) - 1:
                size = prediction.size()
                pred_img = prediction.view_class(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                gt_img = spectrogram.view_class(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                train_prediction_img = np.concatenate([pred_img, gt_img], axis=-1)
                train_prediction_img = map_numpy_values(train_prediction_img, (0, 1), current_range=(-1, 1))

        # validation loop
        for i, data_pair in enumerate(validation_dataset_loader):
            # get batch data
            metadata, _, spectrogram, coordinates = data_pair
            spectrogram_shape = spectrogram.size()
            spectrogram = spectrogram.reshape((spectrogram_shape[0], spectrogram_shape[1], spectrogram_shape[2], 1))

            # tensors to device
            spectrogram = spectrogram.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            with torch.no_grad():
                prediction = model(coordinates, metadata)
                prediction_shape = spectrogram.size()
                prediction = prediction.reshape((prediction_shape[0], prediction_shape[1], prediction_shape[2], 1))
                loss, individual_losses = criterion(prediction, spectrogram)
                individual_losses['loss'] = loss
                individual_losses = {key: value.item() for key, value in individual_losses.items()}
                eval_losses.append(individual_losses)

            # image for tensorboard
            if i == len(validation_dataset_loader) - 1:
                size = prediction.size()
                pred_img = prediction.view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                gt_img = spectrogram.view(1, 1, 1, size[1], size[2]).detach().cpu().numpy()
                eval_prediction_img = np.concatenate([pred_img, gt_img], axis=-1)
                eval_prediction_img = map_numpy_values(eval_prediction_img, (0, 1), current_range=(-1, 1))

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


if __name__ == "__main__":
    # ray config
    num_trials = 20
    max_num_epochs = 100
    gpus_per_trial = 1

    # config
    config = {
        # data
        "training_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/training")),
        "validation_dataset_path": os.path.normpath(os.path.join(os.getcwd(), "Dataset/validation")),
        "num_mfccs": 20,  # tune.choice([20, 50, 128]),
        "feature_mapping_file": os.path.normpath(os.getcwd() + "/data_handling/feature_mapping.json"),
        'transformation_file': os.path.normpath(os.getcwd() + "/Dataset/transformation.json"),

        # data loading
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": 384,  # tune.choice([128, 256, 384, 512]),
        "SIREN_hidden_layers": 3,  # tune.choice([3, 5, 8]),
        "SIREN_use_harmonic_embedding": True,
        "SIREN_num_harmonic_functions": 50,  # tune.choice([30, 50, 80, 100]),
        "MODULATION_Type": MappingType.Mult_Networks_One_Dimension_For_Each_Layer,  # tune.choice(list(MappingType)),
        "MODULATION_hidden_features": 384,  # tune.choice([128, 256, 384, 512]),
        "MODULATION_hidden_layers": 8,  # tune.choice([3, 5, 8]),
        "MODULATION_use_harmonic_embedding": True,  # tune.choice([True, False]),
        "MODULATION_num_harmonic_functions": 2,  # tune.choice([2, 4, 6]),

        # training
        "lr": 5e-05, #tune.choice([0.00005, 0.000075, 0.0001]),
        "epochs": 100,
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
        grace_period=100,
        reduction_factor=1.5)
    reporter = CLIReporter(
        metric_columns=["eval_loss", "training_iteration"])
    experiment_datetime = datetime.now()
    print(experiment_datetime)
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

    # analysis
    checkpoint_dir = Path('./Checkpoints')
    experiment_path = nearest_experiment_path(checkpoint_dir, experiment_datetime)
    experiment_name = os.path.basename(experiment_path)
    default_excluded_fields = ['eval_vid', 'train_vid', 'time_this_iter_s', 'done', 'timesteps_total',
                               'episodes_total', 'trial_id', 'experiment_id', 'date', 'timestamp', 'pid', 'hostname',
                               'node_ip', 'config', 'time_since_restore', 'timesteps_since_restore',
                               'iterations_since_restore', 'warmup_time', 'transformation_file',
                               'training_dataset_path', 'validation_dataset_path', 'feature_mapping_file']

    analysis = RayTuneAnalysis(checkpoint_dir, experiment_names=[experiment_name],
                               excluded_fields=default_excluded_fields)
    table_view = TextTableView(analysis)

    text = table_view.statistics_view + '\n\n\n' + table_view.config_analysis_view + '\n\n\n'
    text += 'Best Trial Path: ' + best_trial_path(experiment_path)

    analysis_file = os.path.join(experiment_path, 'analysis.txt')
    write_textfile(text, analysis_file)
