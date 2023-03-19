import os

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

from data_handling import util


def get_padded_audio_file(trial_file,  output_length=96000):
    audio_samples, sample_rate = librosa.load(
        trial_file, sr=librosa.get_samplerate(trial_file), mono=True)

    if len(audio_samples) > output_length:
        center_lower_bound = len(audio_samples) // 2 - output_length // 2
        center_upper_bound = len(audio_samples) // 2 + output_length // 2
        audio_samples = audio_samples[center_lower_bound:center_upper_bound]
    else:
        audio_samples = librosa.util.pad_center(
            audio_samples, size=output_length, mode='constant'
        )

    return audio_samples, sample_rate


def pad_audio_files(samples_directory, output_length=96000):
    """Pads audio files in samples_directory to output_length.
    
    samples_directory: Directory in which files are in

    output_length: Length of signal to pad to
    """
    audio_files = util.files_in_directory(
        samples_directory, file_patterns=["**/*.wav", "**/*.flac"], recursive=True)

    print('Preprocessing pad: Found {} Audio Files'.format(len(audio_files)))

    for file in tqdm(audio_files, desc='Pad audio files', unit='Audio Files'):
        audio_samples, sample_rate = get_padded_audio_file(file, output_length=output_length)

        soundfile.write(file, audio_samples, samplerate=sample_rate)
