import os
import librosa

from data_handling import util
from tqdm import tqdm
import numpy as np


def generate_spectogram_for_trial(trial_file):
    audio_samples, audio_sampling_rate = librosa.load(
        trial_file, sr=librosa.get_samplerate(trial_file), mono=True)
    
    spectogram = librosa.feature.melspectrogram(y=audio_samples, sr=audio_sampling_rate)
    return np.transpose(spectogram, (1, 0))

def save_spectogram_for_trials(samples_directory):
    audio_files = util.files_in_directory(
        samples_directory, file_patterns=["**/*.wav", "**/*.flac"], recursive=True)

    print('Found {} Audio Files'.format(len(audio_files)))

    for file in tqdm(audio_files):
        trial_name = os.path.splitext(os.path.basename(file))[0]
        
        spectogram_file = os.path.join(os.path.dirname(
            file), '{}_spectogram.npy'.format(trial_name))

        spectogram = generate_spectogram_for_trial(file)

        np.save(spectogram_file, spectogram)