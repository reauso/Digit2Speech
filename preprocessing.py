import os
import argparse

from data_handling.preprocessing.split_audio_signals import split_audio
from data_handling.preprocessing.train_val_split import split_train_val
from data_handling.preprocessing.mfcc import save_mfcc_for_trials
from data_handling.preprocessing.spectrogram import save_spectrogram_for_trials

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_audio", action="store_true")
    parser.add_argument("--split_train_val", action="store_true")
    parser.add_argument("--mfcc", action="store_true")
    parser.add_argument("--spectrogram", action="store_true")
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
        save_mfcc_for_trials(training_directory)
        save_mfcc_for_trials(validation_directory)

    if args.spectrogram:
        save_spectrogram_for_trials(training_directory)
        save_spectrogram_for_trials(validation_directory)
