import argparse
import os

from data_handling.preprocessing.mfcc import save_mfcc_for_trials
from data_handling.preprocessing.pad_signal import pad_audio_files
from data_handling.preprocessing.spectrogram import save_spectrogram_for_trials
from data_handling.preprocessing.split_audio_signals import split_audio
from data_handling.preprocessing.train_val_split import split_train_val
from data_handling.preprocessing.transform import save_transform_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_audio", action="store_true", help='Splits the Audio Captures into several files.')
    parser.add_argument("--split_train_val", action="store_true", help='Splits a distribution of files into training'
                                                                       'and validation datasets.')
    parser.add_argument("--mfcc", action="store_true", help='Creates mfcc coefficients for the Datasets.')
    parser.add_argument("--num_mfcc", type=int, default=50, help='The number of mfcc coefficients to generate.')
    parser.add_argument("--spectrogram", action="store_true", help='Creates Spectrogram images for the Datasets.')
    parser.add_argument("--transform", action="store_true", help='Creates a transformation file for the signal so that'
                                                                 'the dataset can cover optimal value range.')
    parser.add_argument("--pad", action="store_true", help='Applies Zero Padding to Audio files which are less than '
                                                           '2 sec long.')
    args = parser.parse_args()

    # define folder paths
    dataset_folder = os.path.join(os.getcwd(), "Dataset")
    raw_samples_directory = os.path.join(dataset_folder, 'raw-samples')
    samples_directory = os.path.join(dataset_folder, 'samples')
    training_directory = os.path.join(dataset_folder, "training")
    validation_directory = os.path.join(dataset_folder, "validation")
    transform_path = os.path.join(dataset_folder, "transformation.json")

    if args.split_audio:
        split_audio(raw_samples_directory, samples_directory)

    if args.split_train_val:
        split_train_val(samples_directory, training_directory,
                        validation_directory)

    if args.pad:
        pad_audio_files(training_directory)
        pad_audio_files(validation_directory)

    if args.mfcc:
        save_mfcc_for_trials(training_directory, n_mfcc=args.num_mfcc)
        save_mfcc_for_trials(validation_directory, n_mfcc=args.num_mfcc)

    if args.spectrogram:
        save_spectrogram_for_trials(training_directory)
        save_spectrogram_for_trials(validation_directory)

    if args.transform:
        save_transform_data(
            transform_path, training_directory, validation_directory)
