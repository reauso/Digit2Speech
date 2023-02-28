import os
import librosa
import librosa.display
import cv2

from data_handling import util
from tqdm import tqdm
import numpy as np

from data_handling.util import normalize_numpy


def save_spectrogram_for_trials(samples_directory):
    audio_files = util.files_in_directory(
        samples_directory, file_patterns=["**/*.wav", "**/*.flac"], recursive=True)

    print('Found {} Audio Files'.format(len(audio_files)))

    for file in tqdm(audio_files, desc='Generate Mel Spectrogram\'s', unit='Audio Files'):
        trial_name = os.path.splitext(os.path.basename(file))[0]

        spectrogram_file = os.path.join(os.path.dirname(file), '{}_spectrogram.png'.format(trial_name))
        audio_samples, audio_sampling_rate = librosa.load(file, sr=librosa.get_samplerate(file), mono=True)

        spectrogram = librosa.feature.melspectrogram(y=audio_samples, sr=audio_sampling_rate)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram = normalize_numpy(spectrogram, (0.0, 255.0), current_range=(-80.0, 0.0))

        cv2.imwrite(spectrogram_file, spectrogram)
