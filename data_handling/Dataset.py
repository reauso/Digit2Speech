import json
import os

import librosa
import numpy as np
import torch

from data_handling.util import files_in_directory, get_metadata_from_file_name, normalize_tensor


class DigitAudioDataset(torch.utils.data.Dataset):
    def __init__(self, path, audio_sample_coverage=1.0, shuffle_audio_samples=True, num_mfcc=50,
                 feature_mapping_file=os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')):
        # validate
        if not 1.0 >= audio_sample_coverage > 0.0:
            raise ValueError('sample_coverage ranges between (0;1]! Given value was {}.'.format(audio_sample_coverage))
        if not os.path.exists(feature_mapping_file):
            raise FileNotFoundError('The feature mapping file does not exists: {}'.format(feature_mapping_file))

        # set parameters
        self.path = path
        self.audio_sample_coverage = audio_sample_coverage
        self.shuffle_audio_samples = shuffle_audio_samples
        self.num_mfcc = num_mfcc

        # necessary values
        self.data_pair_base_paths = []
        self.data_pair_audio_extensions = []

        # load feature mappings
        f = open(feature_mapping_file)
        feature_mappings = json.load(f)
        self.speaker_sex_mapping = feature_mappings['speakers-sex']
        self.language_mapping = feature_mappings['language-index']
        self.sex_mapping = feature_mappings['sex-index']
        f.close()

        # check for valid data pairs in path
        self.audio_files = files_in_directory(self.path, ['**/*.wav', "**/*.flac"], recursive=True)
        for file in self.audio_files:
            data_pair_name = os.path.splitext(os.path.basename(file))[0]
            data_pair_audio_extension = os.path.splitext(os.path.basename(file))[1]
            data_pair_files_dir = os.path.dirname(file)
            data_pair_base_path = os.path.join(data_pair_files_dir, data_pair_name)

            # continue if no mfcc file exists
            mfcc_file_path = '{}_mfcc_{}.npy'.format(data_pair_base_path, self.num_mfcc)
            if not os.path.exists(mfcc_file_path):
                continue

            self.data_pair_base_paths.append(data_pair_base_path)
            self.data_pair_audio_extensions.append(data_pair_audio_extension)

        print('Found {} Data Pairs for Dataset'.format(len(self.data_pair_base_paths)))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Loads the data pair with the corresponding id.
        :param idx: The data Pair to load.
        :return: Returns the Metadata object, (len(audio samples) * sample_coverage) random unique audio samples
        from the audio file and an array with the index (time) of the random unique audio samples in the
        original array.
        """
        data_pair_base_path = self.data_pair_base_paths[idx]
        audiofile_extension = self.data_pair_audio_extensions[idx]
        print(data_pair_base_path)

        # get metadata values of current data pair as int
        language, speaker, digit, _ = get_metadata_from_file_name(data_pair_base_path)
        language = self.language_mapping[language]
        sex = self.sex_mapping[self.speaker_sex_mapping[speaker]]
        digit = int(digit)

        # metadata to normalized float tensor
        language = normalize_tensor(torch.FloatTensor([language]), min_value=0,
                                    max_value=len(self.language_mapping) - 1)
        sex = normalize_tensor(torch.FloatTensor([sex]), min_value=0, max_value=len(self.sex_mapping) - 1)
        digit = normalize_tensor(torch.FloatTensor([digit]), min_value=0, max_value=9)

        # load audio file
        audio_file = '{}{}'.format(data_pair_base_path, audiofile_extension)
        signal, sample_rate = librosa.load(audio_file, sr=librosa.get_samplerate(audio_file), mono=True)

        # load mfcc file
        mfcc_file = '{}_mfcc_{}.npy'.format(data_pair_base_path, self.num_mfcc)
        mfcc_coefficients = np.load(mfcc_file).reshape(self.num_mfcc)
        # spectogram_file = '{}_spectogram.npy'.format(data_pair_base_path)
        # spectogram = np.load(spectogram_file)

        # create metadata object
        metadata = {
            "language": language,
            "digit": digit,
            "sex": sex,
            "mfcc_coefficients": mfcc_coefficients,
            # "spectogram": spectogram
        }

        # random audio sampling
        num_total_samples = len(signal)
        num_covered_samples = int(num_total_samples * self.audio_sample_coverage)
        num_covered_samples = num_covered_samples if num_covered_samples > 0 else 1  # minimum 1 sample

        random_audio_sample_indices = np.random.choice(num_total_samples, size=num_covered_samples, replace=False)
        if not self.shuffle_audio_samples:
            random_audio_sample_indices = np.sort(random_audio_sample_indices, kind='quicksort')

        random_audio_samples = signal[random_audio_sample_indices]

        # return
        return metadata, torch.FloatTensor(random_audio_samples), torch.FloatTensor(random_audio_sample_indices)


if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'Dataset', 'samples')
    dataset = DigitAudioDataset(data_path, audio_sample_coverage=0.2, shuffle_audio_samples=False)

    data_pair = dataset[0]
    print(data_pair[0])
    print(data_pair[1])
    print(data_pair[2])
    print(len(data_pair[1]))
    print(len(data_pair[2]))
    print(len(np.unique(data_pair[2])))
