import os
import librosa
import numpy as np

import util.data_helper
from tqdm import tqdm


def calculate_mfcc_for_trial(trial_file,  n_mfcc=50):
    audio_samples, audio_sampling_rate = librosa.load(
        trial_file, sr=librosa.get_samplerate(trial_file), mono=True)

    # best for speech processing
    n_fft = int((audio_sampling_rate / 22050) * 512)

    hop_length = len(audio_samples)+1
    mfcc_coefficients = librosa.feature.mfcc(y=audio_samples, sr=audio_sampling_rate, n_mfcc=n_mfcc,
                                             n_fft=n_fft, hop_length=hop_length)

    return np.transpose(mfcc_coefficients, (1, 0))


def save_mfcc_for_trials(samples_directory, n_mfcc=50):
    audio_files = util.data_helper.files_in_directory(
        samples_directory, file_patterns=["**/*.wav", "**/*.flac"], recursive=True)

    print('Found {} Audio Files'.format(len(audio_files)))

    for file in tqdm(audio_files, desc='Generate MFCC\'s', unit='Audio Files'):
        trial_name = os.path.splitext(os.path.basename(file))[0]

        mfcc_file = os.path.join(os.path.dirname(
            file), '{}_mfcc_{}.npy'.format(trial_name, n_mfcc))

        mfcc_coefficients = calculate_mfcc_for_trial(file, n_mfcc=n_mfcc)

        np.save(mfcc_file, mfcc_coefficients)
