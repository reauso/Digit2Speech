import json
import os
import cv2

import librosa
import numpy as np
import soundfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_handling.Dataset import DigitAudioDatasetForSpectrograms
from data_handling.util import read_textfile, files_in_directory, get_metadata_from_file_name, normalize_tensor, \
    object_to_float_tensor, map_numpy_values
from model.SirenModel import SirenModelWithFiLM

if __name__ == "__main__":
    # define values
    sample_rate = 48000
    seconds = 2
    use_metadata_from_file = True
    language = 'english'
    sex = 'female'
    digit = '0'

    # define necessary paths
    source_path = os.path.join(os.getcwd(), os.path.normpath('Dataset/validation'))
    save_path = os.path.join(os.getcwd(), 'GeneratedAudio')
    experiment_name = 'train_2023-03-06_05-55-48'
    trial_name = 'train_29351_00023_23_SIREN_hidden_features=256,SIREN_hidden_layers=3,lr=0.0002_2023-03-06_06-38-23'
    model_path = os.path.join(os.getcwd(), 'Checkpoints', experiment_name, trial_name)
    feature_mapping_file = os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')

    # create save dir
    os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)

    # load feature mappings
    f = open(feature_mapping_file)
    feature_mappings = json.load(f)
    speaker_sex_mapping = feature_mappings['speakers-sex']
    f.close()

    # create model instance
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_config = json.loads(read_textfile(os.path.join(model_path, 'params.json')))
    model = SirenModelWithFiLM(in_features=2,  # x coord and y coord
                               out_features=1,  # grayscale value of spectrogram at (x,y) coord
                               hidden_features=model_config["SIREN_hidden_features"],
                               hidden_layers=model_config["SIREN_hidden_layers"],
                               mod_features=model_config['num_mfccs'] * 4)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # create dataset
    validation_dataset = DigitAudioDatasetForSpectrograms(
        path=source_path,
        num_mfcc=model_config['num_mfccs'],
        feature_mapping_file=feature_mapping_file,
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

        # tensors to device
        coordinates = coordinates.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        with torch.no_grad():
            filename_image = '{}-lang-{}-sex-{}-digit-{}.png'.format(i, raw_metadata['language'], raw_metadata['sex'],
                                                                     raw_metadata['digit'])
            filename_audio = '{}-lang-{}-sex-{}-digit-{}.wav'.format(i, raw_metadata['language'], raw_metadata['sex'],
                                                                     raw_metadata['digit'])
            filepath_image = os.path.join(save_path, experiment_name, filename_image)
            filepath_audio = os.path.join(save_path, experiment_name, filename_audio)

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
            signal = librosa.feature.inverse.mel_to_audio(spectrogram, sr=48000, hop_length=512, length=96000)
            soundfile.write(filepath_audio, signal, 48000)
