import os
import librosa

from data_handling import util
from tqdm import tqdm
import numpy as np


def generate_spectrogram_for_trial(trial_file):
    audio_samples, audio_sampling_rate = librosa.load(
        trial_file, sr=librosa.get_samplerate(trial_file), mono=True)
    
    spectrogram = librosa.feature.melspectrogram(y=audio_samples, sr=audio_sampling_rate)
    return np.transpose(spectrogram, (1, 0))


def save_spectrogram_for_trials(samples_directory):
    audio_files = util.files_in_directory(
        samples_directory, file_patterns=["**/*.wav", "**/*.flac"], recursive=True)

    print('Found {} Audio Files'.format(len(audio_files)))

    for file in tqdm(audio_files):
        trial_name = os.path.splitext(os.path.basename(file))[0]
        
        spectrogram_file = os.path.join(os.path.dirname(
            file), '{}_spectrogram.npy'.format(trial_name))

        spectrogram = generate_spectrogram_for_trial(file)

        np.save(spectrogram_file, spectrogram)
