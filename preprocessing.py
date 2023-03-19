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
    parser.description = "Script to perform preprocessing steps on data. Expects data to be in `Dataset/samples` and outputs into `Dataset/training` and `Dataset/validation`"
    
    parser.add_argument("--split_audio", help="Controls if audio files should be analyzed for relevant signal. DO NOT USE IF THE VALID SAMPLES ARE USED.",
                        action="store_true")
    parser.add_argument("--split_train_val", help="Controls if all files should be split into training and validation.",
                        action="store_true")
    parser.add_argument("--mfcc", help="Controls if MFCC coefficients should be generated",
                        action="store_true")
    parser.add_argument("--spectrogram", help="Controls if mel spectrograms should be generated",
                        action="store_true")
    parser.add_argument("--transform", action="store_true")
    parser.add_argument("--pad", help="Controls if audio files should be padded to 2 seconds",
                        action="store_true")
    args = parser.parse_args()
    
    if not any(args.__dict__.values()):
        parser.print_help()
        exit()

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
        save_mfcc_for_trials(training_directory, n_mfcc=50)
        save_mfcc_for_trials(validation_directory, n_mfcc=50)

    if args.spectrogram:
        save_spectrogram_for_trials(training_directory)
        save_spectrogram_for_trials(validation_directory)

    if args.transform:
        save_transform_data(
            transform_path, training_directory, validation_directory)
