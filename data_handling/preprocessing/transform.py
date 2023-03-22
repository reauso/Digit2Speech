import json

import librosa
import numpy as np
from tqdm import tqdm

from util.data_helper import files_in_directory


def save_transform_data(transform_path, training_directory=None, validation_directory=None):
    all_audio_files = []

    if training_directory is not None:
        all_audio_files.extend(files_in_directory(training_directory, ['**/*.wav', "**/*.flac"], recursive=True))
    if training_directory is not None:
        all_audio_files.extend(files_in_directory(validation_directory, ['**/*.wav', "**/*.flac"], recursive=True))

    # define numpy arrays to save collected data of each file
    all_min_signal_values = np.zeros((len(all_audio_files)), dtype=float)
    all_max_signal_values = np.zeros((len(all_audio_files)), dtype=float)

    # collect all min and max values
    pipeline = tqdm(all_audio_files, unit='Audio Files', desc='Get Min/Max Value')
    for i, audio_file in enumerate(pipeline):
        signal, _ = librosa.load(audio_file, sr=librosa.get_samplerate(audio_file), mono=True)

        all_min_signal_values[i] = np.min(signal)
        all_max_signal_values[i] = np.max(signal)

    print('Estimated Total Min Value: {} and Max Value: {}.'.format(np.min(all_min_signal_values), np.max(all_max_signal_values)))
    print('Min File Name: {}'.format(all_audio_files[np.argmin(all_min_signal_values)]))
    print('Max File Name: {}'.format(all_audio_files[np.argmax(all_max_signal_values)]))
    print('Min Metadata: min: {:.5f}, max: {:.5f}, mean: {:.5f}, std: {:.5f}'.format(np.min(all_min_signal_values), np.max(all_min_signal_values), np.mean(all_min_signal_values), np.std(all_min_signal_values)))
    print('Max Metadata: min: {:.5f}, max: {:.5f}, mean: {:.5f}, std: {:.5f}'.format(np.min(all_max_signal_values), np.max(all_max_signal_values), np.mean(all_max_signal_values), np.std(all_max_signal_values)))

    min_signal_value = np.mean(all_min_signal_values) - np.std(all_min_signal_values)
    max_signal_value = np.mean(all_max_signal_values) + np.std(all_max_signal_values)
    print('Use Min Value: {} and Max Value: {} for Scale and Shift Calculation.'.format(min_signal_value, max_signal_value))

    abs_difference = min_signal_value + max_signal_value
    shift = abs_difference / 2

    delta = max_signal_value - min_signal_value
    delta = delta if delta != 0 else 0.00000000001
    scale = 2 / delta

    print('Estimated Shift: {} and Scale: {}.'.format(shift, scale))

    transform_dict = {
        'shift': shift,
        'scale': scale,
    }

    f = open(transform_path, 'w')
    json.dump(transform_dict, f)
