import os
import time
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from torch.utils.data import DataLoader

from data_handling.Dataset import DigitAudioDataset
from model.SirenModel import SirenModelWithFiLM


def train(config, checkpoint_dir=None):
    torch.multiprocessing.set_start_method('spawn')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # create datasets and data loaders
    train_dataset = DigitAudioDataset(
        path=config['training_dataset_path'],
        audio_sample_coverage=config['audio_sample_coverage'].sample(),
        shuffle_audio_samples=config['shuffle_audio_samples'],
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    train_dataset_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                      shuffle=config['shuffle_audio_files'], num_workers=0, drop_last=False)

    '''validation_dataset = DigitAudioDataset(
        path=config['validation_dataset_path'],
        audio_sample_coverage=1.0,
        shuffle_audio_samples=False,
        num_mfcc=config['num_mfccs'],
        feature_mapping_file=config['feature_mapping_file'],
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=config['batch_size'],
                                           shuffle=False, num_workers=0, drop_last=False)'''

    # create model
    model = SirenModelWithFiLM(in_features=1,
                               hidden_features=config["SIREN_hidden_features"].sample(),
                               num_layers=config["SIREN_num_layers"].sample(),
                               out_features=1,
                               mod_features=config["SIREN_mod_features"].sample())
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # load from checkpoint if checkpoint set
    if checkpoint_dir:
        load_file_path = os.path.join(checkpoint_dir, "checkpoint")
        print("Load from Checkpoint: {}".format(load_file_path))
        model_state, optimizer_state = torch.load(load_file_path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # necessary values and objects for training loop

    # training loop
    for epoch in range(config["epochs"]):
        for batch in train_dataset_loader:
            # get batch data
            metadata, audio_samples, audio_sample_indices = batch
            audio_samples = audio_samples.to(device)
            audio_sample_indices = audio_sample_indices.to(device)

            # convert metadata into one array with floats values
            modulation_input = torch.cat([
                metadata['language'],
                metadata['digit'],
                metadata['sex'],
                metadata['mfcc_coefficients']], dim=1)

            # zero gradients
            optimizer.zero_grad()

            # get prediction
            prediction = model()  # TODO hier weitermachen

        exit()
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):

            # Important that batchsize is one
            coords = data["coords"][0].to(device)
            exp = data["exp"][0].to(device)
            pose = data["pose"][0].to(device)
            mask = data["mask"][0].to(device)
            gt_pixels = data["img"][0].to(device)

            # forward + backward + optimize
            x_input = torch.cat([coords], dim=1)
            if config["modulation_type"] == "ExpPose":
                mod = torch.cat([exp[:, :], pose[:, :]], dim=1)
            elif config["modulation_type"] == "DeepSpeech":
                dsf = data["dsf"][0].to(device)
                mod = dsf
            # with torch.cuda.amp.autocast():
            pred_pixels = model(x=x_input, modulation_input=mod[:, :config["SIREN_mod_features"]])

            pred_img = pred_pixels.permute(1, 0).view(1, 3, 256, 256)
            gt_img = gt_pixels.permute(1, 0).view(1, 3, 256, 256)
            loss, _, _ = criterion(prediction=pred_img, ground_truth=gt_img)  # TODO ersetzen durch unsere Loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

            if config["short_gpu_sleep"]:
                time.sleep(0.0175)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                # Important that batchsize is one
                coords = data["coords"][0].to(device)
                exp = data["exp"][0].to(device)
                pose = data["pose"][0].to(device)
                mask = data["mask"][0].to(device)
                gt_pixels = data["img"][0].to(device)

                x_input = torch.cat([coords], dim=1)
                mod = torch.cat([exp[:, :], pose[:, :]], dim=1)
                # with torch.cuda.amp.autocast():
                if config["modulation_type"] == "ExpPose":
                    mod = torch.cat([exp[:, :], pose[:, :]], dim=1)
                elif config["modulation_type"] == "DeepSpeech":
                    mod = dsf
                pred_pixels = model(x=x_input, modulation_input=mod[:, :config["SIREN_mod_features"]])

                pred_img = pred_pixels.permute(1, 0).view(1, 3, 256, 256)
                gt_img = gt_pixels.permute(1, 0).view(1, 3, 256, 256)
                loss, loss_dict, raw_loss_dict = criterion(prediction=pred_img, ground_truth=gt_img)

                val_loss += loss.cpu().numpy()
                val_steps += 1
            break  # TODO

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # tune?
        pred_img = pred_pixels.permute(1, 0).view(1, 1, 3, 256,
                                                  256).detach().cpu().numpy()  # np.random.randn(1, 1, 3, 10, 10)
        gt_img = gt_pixels.permute(1, 0).view(1, 1, 3, 256, 256).detach().cpu().numpy()
        img_np = np.concatenate([pred_img, gt_img], axis=-1)

        metrics = {
            "loss": (val_loss / val_steps),
            "vid": img_np
        }
        # #TODO Update to mean of all vals insted of last val sample
        for _loss_type, _loss_value in loss_dict.items():
            loss_dict[_loss_type] = _loss_value.item()
        for _loss_type, _loss_value in raw_loss_dict.items():
            loss_dict[_loss_type] = _loss_value.item()

        metrics.update(loss_dict)
        tune.report(**metrics)
        # tune.report(loss=(val_loss / val_steps), vid=img_np)#accuracy=correct / total)
    print("Finished Training")


if __name__ == "__main__":
    config = {
        # data
        "training_dataset_path": os.path.normpath("Dataset/samples"),
        "validation_dataset_path": os.path.normpath("Dataset/validation"),
        "audio_sample_coverage": tune.quniform(0.2, 0.5, 0.1),
        "shuffle_audio_samples": tune.choice([True, False]),
        "num_mfccs": 50,
        "feature_mapping_file": os.path.normpath("data_handling/feature_mapping.json"),

        # data loading
        'batch_size': 1,
        'shuffle_audio_files': True,

        # model
        "SIREN_hidden_features": tune.choice([128]),
        "SIREN_num_layers": tune.choice([5]),
        "SIREN_mod_features": tune.choice([348]),

        # training
        "lr": 0.001,
        "epochs": 50,
    }

    train(config)
