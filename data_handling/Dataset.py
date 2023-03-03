import json
import os

import librosa
import cv2
import numpy as np
import torch

from data_handling.util import files_in_directory, get_metadata_from_file_name, normalize_tensor, read_textfile, \
    object_to_float_tensor


class DigitAudioDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_mfcc=50,
                 feature_mapping_file=os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')):
        # validate
        if not os.path.exists(feature_mapping_file):
            raise FileNotFoundError('The feature mapping file does not exists: {}'.format(feature_mapping_file))

        # set parameters
        self.path = path
        self.num_mfcc = num_mfcc

        # load feature mappings
        f = open(feature_mapping_file)
        feature_mappings = json.load(f)
        f.close()
        self.speaker_sex_mapping = feature_mappings['speakers-sex']

        # necessary values
        self.mfcc_file_pattern = '{}_mfcc_{}.npy'.format('{}', self.num_mfcc)

        # get initial data pairs from existing audio files
        audio_files = files_in_directory(self.path, ['**/*.wav', "**/*.flac"], recursive=True)
        base_names = [os.path.splitext(os.path.basename(file))[0] for file in audio_files]
        base_paths = [os.path.dirname(file) for file in audio_files]
        self.data_pairs = [{
            'base_name': base_names[i],
            'base_path': os.path.join(base_paths[i], base_names[i]),
            'audio_extension': os.path.splitext(os.path.basename(file))[1],
        } for i, file in enumerate(audio_files)]

        # check for valid data pairs
        self._filter_for_valid_data_pairs()

    def _filter_for_valid_data_pairs(self):
        self.data_pairs = [
            pair for pair in self.data_pairs if os.path.exists(self.mfcc_file_pattern.format(pair['base_path']))]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        Loads the data pair with the corresponding id.
        :param idx: The data Pair to load.
        :return: Returns the Metadata object of the data pair
        """
        current_pair = self.data_pairs[idx]

        # get metadata values of current data pair
        language, speaker, digit, _ = get_metadata_from_file_name(current_pair['base_name'])
        sex = self.speaker_sex_mapping[speaker]

        # metadata to normalized float tensor
        language = object_to_float_tensor(language, self.num_mfcc)
        sex = object_to_float_tensor(sex, self.num_mfcc)
        digit = object_to_float_tensor(digit, self.num_mfcc)

        # load mfcc file
        mfcc_file = self.mfcc_file_pattern.format(current_pair['base_path'])
        mfcc_coefficients = np.load(mfcc_file).reshape(self.num_mfcc)
        mfcc_coefficients = torch.FloatTensor(mfcc_coefficients)

        # create metadata object
        metadata = {
            "language": language,
            "digit": digit,
            "sex": sex,
            "mfcc_coefficients": mfcc_coefficients,
        }

        # return
        return metadata


class DigitAudioDatasetForSignal(DigitAudioDataset):
    def __init__(self, path, audio_sample_coverage=1.0, shuffle_audio_samples=True, num_mfcc=50,
                 feature_mapping_file=os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json'),
                 transformation_file=None):
        super().__init__(
            path=path,
            num_mfcc=num_mfcc,
            feature_mapping_file=feature_mapping_file
        )

        if not 1.0 >= audio_sample_coverage > 0.0:
            raise ValueError('sample_coverage ranges between (0;1]! Given value was {}.'.format(audio_sample_coverage))
        if transformation_file is not None and not os.path.exists(transformation_file):
            raise FileNotFoundError('The transformation file does not exists: {}'.format(feature_mapping_file))

        # set parameters
        self.audio_sample_coverage = audio_sample_coverage
        self.shuffle_audio_samples = shuffle_audio_samples

        # load transformation
        if transformation_file is not None:
            raw_transformation_text = read_textfile(transformation_file)
            transformation = json.loads(raw_transformation_text)
            self.shift = transformation['shift']
            self.scale = transformation['scale']
        else:
            self.shift = 0.0
            self.scale = 1.0

        print('Found {} Data Pairs for Dataset at {}'.format(len(self.data_pairs), self.path))
        print('Use Shift: {} and Scale: {}.'.format(self.shift, self.scale))

    def __getitem__(self, idx):
        """
        Loads the data pair with the corresponding id.
        :param idx: The data Pair to load.
        :return: Returns the Metadata object, (len(audio samples) * sample_coverage) random unique audio samples
        from the audio file and an array with the index (time) of the random unique audio samples in the
        original array.
        """
        current_pair = self.data_pairs[idx]

        # get metadata of current_pair object
        metadata = super().__getitem__(idx)

        # load audio file
        audio_file = '{}{}'.format(current_pair['base_path'], current_pair['audio_extension'])
        signal, _ = librosa.load(audio_file, sr=librosa.get_samplerate(audio_file), mono=True)

        # apply shift and scale to signal
        signal = (signal - self.shift) * self.scale

        # random audio sampling
        num_total_samples = len(signal)
        num_covered_samples = int(num_total_samples * self.audio_sample_coverage)
        num_covered_samples = num_covered_samples if num_covered_samples > 0 else 1  # minimum 1 sample

        random_audio_sample_indices = np.random.choice(num_total_samples, size=num_covered_samples, replace=False)
        if not self.shuffle_audio_samples:
            random_audio_sample_indices = np.sort(random_audio_sample_indices, kind='quicksort')

        random_audio_samples = signal[random_audio_sample_indices]

        return metadata, torch.FloatTensor(random_audio_samples), torch.FloatTensor(random_audio_sample_indices)


class DigitAudioDatasetForSpectrograms(DigitAudioDataset):
    def __init__(self, path, num_mfcc=50, feature_mapping_file=os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')):
        super().__init__(
            path=path,
            num_mfcc=num_mfcc,
            feature_mapping_file=feature_mapping_file
        )

        print('Found {} Data Pairs for Dataset at {}'.format(len(self.data_pairs), self.path))

    def _filter_for_valid_data_pairs(self):
        super()._filter_for_valid_data_pairs()

        spectrogram_file_pattern = '{}_spectrogram.png'
        self.data_pairs = [
            pair for pair in self.data_pairs if os.path.exists(spectrogram_file_pattern.format(pair['base_path']))]

    def __getitem__(self, idx):
        current_pair = self.data_pairs[idx]

        # get metadata of current_pair object
        metadata = super().__getitem__(idx)

        # load spectrogram file
        spectrogram_file = '{}_spectrogram.png'.format(current_pair['base_path'])
        spectrogram = cv2.imread(spectrogram_file, cv2.IMREAD_GRAYSCALE)
        spectrogram = torch.FloatTensor(spectrogram)
        spectrogram = normalize_tensor(spectrogram, min_value=0, max_value=255)

        return metadata, spectrogram


if __name__ == '__main__':
    data_path = os.path.abspath('./Dataset/training')
    dataset = DigitAudioDatasetForSignal(data_path, num_mfcc=20)
    print(dataset[0])
    data_path = os.path.abspath('./Dataset/training')
    dataset = DigitAudioDatasetForSpectrograms(data_path, num_mfcc=20)
    print(dataset[0])
