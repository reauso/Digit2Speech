import argparse
import json
import os
from pathlib import Path

import cv2

import librosa
import numpy as np
import soundfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_handling.Dataset import DigitAudioDatasetForSpectrograms
from data_handling.util import read_textfile, map_numpy_values, latest_experiment_path, best_trial_path
from model.SirenModel import MappingType, SirenModelWithFiLM

if __name__ == "__main__":
    # defaults for config
    checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    source_path = os.path.join(os.getcwd(), os.path.normpath('Dataset/validation'))
    save_path = os.path.join(os.getcwd(), 'GeneratedAudio')
    feature_mapping_file = os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=48000, help='The sample rate of the audio file.')
    parser.add_argument("--seconds", type=float, default=2, help='The seconds to synthesize')
    parser.add_argument("--checkpoint_dir", type=Path, default=Path(checkpoint_dir), help='Dir with all experiments.')
    parser.add_argument("--source_dir", type=Path, default=Path(source_path), help='Source dir with metadata.')
    parser.add_argument("--save_dir", type=Path, default=Path(save_path), help='Saving dir for generated Audio.')
    parser.add_argument("--experiment", type=str, default='latest', help='Name of the experiment of the model or "latest" for automatic detection.')
    parser.add_argument("--trial", type=str, default='best', help='Trial name or "best" for automatic detection')
    parser.add_argument("--feature_mapping_file", type=Path, default=Path(feature_mapping_file),
                        help='The feature mapping file.')
    args = parser.parse_args()

    # automatic detections
    args.experiment = os.path.basename(
        latest_experiment_path(args.checkpoint_dir)) if args.experiment == 'latest' else args.experiment
    experiment_dir = os.path.join(args.checkpoint_dir, args.experiment)
    args.trial = os.path.basename(best_trial_path(experiment_dir)) if args.trial == 'best' else args.trial

    # define necessary paths
    model_path = os.path.join(args.checkpoint_dir, args.experiment, args.trial)
    print('Use Model at location: {}'.format(model_path))

    # create save dir
    os.makedirs(os.path.join(args.save_dir, args.experiment), exist_ok=True)

    # load feature mappings
    f = open(args.feature_mapping_file)
    feature_mappings = json.load(f)
    speaker_sex_mapping = feature_mappings['speakers-sex']
    f.close()

    # create model instance
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_config = json.loads(read_textfile(os.path.join(model_path, 'params.json')))
    model = SirenModelWithFiLM(
        in_features=2,  # x coord and y coord
        out_features=1,  # grayscale value of spectrogram at (x,y) coord
        hidden_features=model_config["SIREN_hidden_features"],
        hidden_layers=model_config["SIREN_hidden_layers"],
        mod_in_features=model_config['num_mfccs'] * 4,
        mod_features=model_config['MODULATION_hidden_features'],
        mod_hidden_layers=model_config['MODULATION_hidden_layers'],
        modulation_type=MappingType(model_config['MODULATION_Type']),
    )
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create dataset
    validation_dataset = DigitAudioDatasetForSpectrograms(
        path=args.source_dir,
        num_mfcc=model_config['num_mfccs'],
        feature_mapping_file=args.feature_mapping_file,
    )
    validation_dataset_loader = DataLoader(validation_dataset, batch_size=1, pin_memory=True, prefetch_factor=10,
                                           shuffle=False, num_workers=4, drop_last=False)

    # load model state
    load_file_path = os.path.join(model_path, "checkpoint")
    print("Load from Checkpoint: {}".format(load_file_path))
    model_state, _ = torch.load(load_file_path)
    model.load_state_dict(model_state)

    # generate audio
    pipeline = tqdm(validation_dataset_loader, unit='Files', desc='Generate Audio Files')
    for i, data_pair in enumerate(pipeline):
        # get batch data
        metadata, raw_metadata, spectrogram, coordinates = data_pair
        spectrogram_shape = spectrogram.size()
        coordinates = coordinates.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        raw_metadata = {key: value[0] for key, value in raw_metadata.items()}

        # tensors to device
        coordinates = coordinates.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        with torch.no_grad():
            filename_image = '{}-lang-{}-sex-{}-digit-{}.png'.format(i, raw_metadata['language'], raw_metadata['sex'],
                                                                     raw_metadata['digit'])
            filename_audio = '{}-lang-{}-sex-{}-digit-{}.wav'.format(i, raw_metadata['language'], raw_metadata['sex'],
                                                                     raw_metadata['digit'])
            filepath_image = os.path.join(args.save_dir, args.experiment, filename_image)
            filepath_audio = os.path.join(args.save_dir, args.experiment, filename_audio)

            # get prediction
            spectrogram = model(coordinates, metadata)
            spectrogram = spectrogram.reshape((spectrogram_shape[1], spectrogram_shape[2]))
            spectrogram = spectrogram.cpu().detach().numpy()

            # save spectrogram image
            spectrogram_image = map_numpy_values(spectrogram, (0.0, 255.0), current_range=(-1.0, 1.0))
            spectrogram_image = np.round(spectrogram_image).astype(np.int32)
            cv2.imwrite(filepath_image, spectrogram_image)

            # generate audio
            spectrogram = map_numpy_values(spectrogram, (-80.0, 0.0), current_range=(-1.0, 1.0))
            spectrogram = librosa.db_to_power(spectrogram)
            signal = librosa.feature.inverse.mel_to_audio(spectrogram, sr=args.sample_rate, hop_length=512,
                                                          length=args.sample_rate * args.seconds)
            soundfile.write(filepath_audio, signal, args.sample_rate)
