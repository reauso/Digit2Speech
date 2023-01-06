import time
import numpy as np
from Datasets.VideoDataset import PixelBasedVideoDataset
from model.SirenModel import SirenModelWithFiLM
import os
import torch
import torch.nn as nn
from torch.utils.data import random_split
from ray import tune
from loss_functions.losses import LossCombination


def train(config, checkpoint_dir=None):

    if config["modulation_type"] == "DeepSpeech":
        config["SIREN_mod_features"] = config["deepspeech_past"] + 1 + config["deepspeech_future"] * 29

    net = SirenModelWithFiLM(in_features=2,
                             hidden_features=config["SIREN_hidden_features"],
                             num_layers=config["SIREN_num_layers"],
                             out_features=3,
                             mod_features=config["SIREN_mod_features"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = LossCombination(l1=config["loss_L1"],
                                mse=config["loss_MSE"],
                                vgg=config["loss_VGG"],
                                ssim=config["loss_SSIM"],
                                masked_mse=config["loss_maskedMSE"]
                                ).compute
    optimizer = torch.optim.Adam(lr=config["lr"], params=net.parameters())

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        print("LOAD FROM CHECKPOINT")

    enable_ds = True if config["modulation_type"] == "DeepSpeech" else False
    trainset = PixelBasedVideoDataset(config["video"], use_deepspeech=enable_ds, ds_past=config["deepspeech_past"], ds_future=config["deepspeech_future"])

    train_subset, val_subset = torch.utils.data.random_split(trainset, [round(len(trainset) * 0.8),
                                                                        round(len(trainset) * 0.2)],
                                                                        generator=torch.Generator().manual_seed(42))

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0)  # IMPORTANT
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=1,
        shuffle=True,
        num_workers=0)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0

        for i, data in enumerate(trainloader):

            # Important that batchsize is one
            coords = data["coords"][0].to(device)
            exp = data["exp"][0].to(device)
            pose = data["pose"][0].to(device)
            mask = data["mask"][0].to(device)
            gt_pixels = data["img"][0].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            x_input = torch.cat([coords], dim=1)
            if config["modulation_type"] == "ExpPose":
                mod = torch.cat([exp[:, :], pose[:, :]], dim=1)
            elif config["modulation_type"] == "DeepSpeech":
                dsf = data["dsf"][0].to(device)
                mod = dsf
            # with torch.cuda.amp.autocast():
            pred_pixels = net(x=x_input, modulation_input=mod[:, :config["SIREN_mod_features"]])

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
                #with torch.cuda.amp.autocast():
                if config["modulation_type"] == "ExpPose":
                    mod = torch.cat([exp[:, :], pose[:, :]], dim=1)
                elif config["modulation_type"] == "DeepSpeech":
                    mod = dsf
                pred_pixels = net(x=x_input, modulation_input=mod[:, :config["SIREN_mod_features"]])

                pred_img = pred_pixels.permute(1, 0).view(1, 3, 256, 256)
                gt_img = gt_pixels.permute(1, 0).view(1, 3, 256, 256)
                loss, loss_dict, raw_loss_dict = criterion(prediction=pred_img, ground_truth=gt_img)

                val_loss += loss.cpu().numpy()
                val_steps += 1
            break  # TODO

            if config["short_gpu_sleep"]:
                time.sleep(0.0175)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        # tune?
        pred_img = pred_pixels.permute(1, 0).view(1, 1, 3, 256, 256).detach().cpu().numpy() # np.random.randn(1, 1, 3, 10, 10)
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
        #tune.report(loss=(val_loss / val_steps), vid=img_np)#accuracy=correct / total)
    print("Finished Training")


if __name__ == "__main__":

    config = {
        "video": "/home/alex/Dropbox/Datasets/PHLA_GreenScreen_Aug_22/PHLA_GreenScreen_Aug_22_trimmed_preprocessed/256/20220823_190124_cut.mp4",

        "SIREN_hidden_features": 128, #tune.sample_from(lambda _: 2 ** np.random.randint(7, 10)),
        "SIREN_num_layers": 5, #tune.choice([5, 6, 7, 8]),
        "SIREN_mod_features": 348, #20, #tune.choice([20]),

        # Film Modulation
        "modulation_type": "DeepSpeech",  # ["ExpPose", "DeepSpeech"]
        # DeepSpeech Settings
        "deepspeech_past": tune.choice([8, 9]),
        "deepspeech_future": tune.choice([8, 9]),

        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": 0.001, #tune.loguniform(1e-4, 1e-1),
        "loss_L1": None,
        "loss_MSE": 40,
        "loss_VGG": 2,
        "loss_SSIM": None,
        "loss_maskedMSE": None,
        "epochs": 1,
        # "batch_size": tune.choice([2, 4, 8, 16])
        "short_gpu_sleep": False
    }
    train(config)
