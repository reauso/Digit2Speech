from data_handling.preprocessing.mfcc import save_mfcc_for_trials
from data_handling import util
import librosa
from data_handling.preprocessing.spectogram import save_spectogram_for_trials
from data_handling.preprocessing.split_audio_signals import trial_midpoint_indices_in_signal
from tqdm import tqdm
import os
import argparse

from data_handling.preprocessing.train_val_split import train_val_split


def split_audio(raw_samples_directory, samples_directory):
    # create output dir if it doesn't exist
    if not os.path.exists(samples_directory):
        os.makedirs(samples_directory)

    # get all audio files
    audio_files = util.files_in_directory(raw_samples_directory)
    print('Found {} Audio Files'.format(len(audio_files)))

    # define necessary values
    threshold = 0.04
    minimal_trial_sample_size = 1000

    for file in tqdm(audio_files):
        # metadata of current file
        language, speaker, digit, _ = util.get_metadata_from_file_name(file)

        # load file
        signal, sample_rate = librosa.load(file, sr=librosa.get_samplerate(file))

        # get midpoint indices of each valid trial in signal
        trial_midpoints = trial_midpoint_indices_in_signal(signal, sample_rate, threshold, minimal_trial_sample_size)

        # save each trial
        for trial_number, trial_mid in enumerate(trial_midpoints):
            trial_signal = signal[trial_mid - sample_rate: trial_mid + sample_rate]
            util.write_trial_to_file(samples_directory, language, speaker, digit, trial_number, trial_signal,
                                     sample_rate)


def split_train_val(samples_directory, training_directory, validation_directory):
    train_val_split(samples_directory, training_directory, validation_directory)


def mfcc(data_directory):
    save_mfcc_for_trials(data_directory)


def spectograms(data_directory):
    save_spectogram_for_trials(data_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_audio", action="store_true")
    parser.add_argument("--split_train_val", action="store_true")
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--spectograms", action="store_true")
    args = parser.parse_args()

    # define folder paths
    dataset_folder = os.path.join(os.getcwd(), "Dataset")
    raw_samples_directory = os.path.join(dataset_folder, 'raw-samples')
    samples_directory = os.path.join(dataset_folder, 'samples')
    training_directory = os.path.join(dataset_folder, "training")
    validation_directory = os.path.join(dataset_folder, "validation")

    if args.split_audio:
        split_audio(raw_samples_directory, samples_directory)

    if args.split_train_val:
        split_train_val(samples_directory, training_directory, validation_directory)

    if args.mfcc:
        mfcc(training_directory)
        mfcc(validation_directory)

    if args.spectograms:
        spectograms(training_directory)
        spectograms(validation_directory)
