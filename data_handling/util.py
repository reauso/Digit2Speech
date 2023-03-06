import glob
import os.path
import random
import threading

import numpy as np
import soundfile
import re
import torch


def files_in_directory(directory_path, file_patterns=None, recursive=False):
    if file_patterns is None:
        file_patterns = ['**']
    elif not isinstance(file_patterns, list):
        file_patterns = [file_patterns]

    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(os.path.join(
            directory_path, pattern), recursive=recursive))

    return files


def read_textfile(textfile_path):
    f = open(textfile_path, 'r')
    text = f.read()
    f.close()

    return text


def get_audio_file_path(folder, language, speaker, digit, trial):
    return os.path.join(folder, f"lang-{language}_speaker-{speaker}_digit-{digit}_trial-{trial}.wav")


def get_metadata_from_file_name(file_path, as_dict=False):
    file_name = os.path.basename(file_path)

    def search_metadata(field, expression, text=file_name):
        return re.search(f"{field}-{expression}", text)

    lang_re = search_metadata("lang", "(\w+)_")
    trial_re = search_metadata("trial", "(\d+)")
    digit_re = search_metadata("digit", "(\d)")
    speaker_re = search_metadata("speaker", "(\d+)")

    language = lang_re.group(1)
    speaker = speaker_re.group(1)
    digit = digit_re.group(1)
    trial = trial_re.group(1)
    if as_dict:
        return {"language": language, "speaker": speaker, "digit": digit, "trial": trial}
    return language, speaker, digit, trial


def write_trial_to_file(output_folder, language, speaker, digit, trial, signal, sample_rate):
    soundfile.write(get_audio_file_path(output_folder, language,
                    speaker, digit, trial), signal, sample_rate)


def map_numpy_values(array, desired_range, current_range=None):
    """
    Normalizes a numpy array to a desired range.
    :param array: The array to map.
    :param desired_range: The desired range.
    :param current_range: The current range. If None is given, then this value is determined
    by numpy's min and max function over the array.
    :return: A Numpy array with arrays values mapped to the desired range.
    """
    if current_range is None:
        min_value = np.min(array)
        max_value = np.max(array)
        current_range = (min_value, max_value)

    current_delta = current_range[1] - current_range[0]
    desired_delta = desired_range[1] - desired_range[0]

    if current_delta <= 0:
        raise ValueError('current_range min is bigger than max')
    if desired_delta <= 0:
        raise ValueError('desired_delta min is bigger than max')

    scale_factor = desired_delta / current_delta
    array = (array - current_range[0]) * scale_factor
    array += desired_range[0]
    return array


def normalize_tensor(tensor, min_value=None, max_value=None):
    """
    Normalizes a tensor to range [-1;1]
    :param tensor: The tensor to normalize.
    :param min_value: The min value to use in mapping calculation. If none is given, then this
    value is determined by torch's min function.
    :param max_value: The max value to use in mapping calculation. If none is given, then this
    value is determined by torch's max function.
    :return: A tensor with values from tensor mapped to [-1;1]
    """
    min_value = min_value if min_value is not None else torch.min(tensor)
    max_value = max_value if max_value is not None else torch.max(tensor)
    delta = max_value - min_value

    if delta == 0:
        return tensor - min_value

    scale_factor = 2 / delta
    tensor = tensor * scale_factor
    tensor = tensor - 1
    return tensor


lock = threading.Lock()


def object_to_float_tensor(obj, tensor_length):
    vec = torch.zeros(tensor_length, dtype=torch.float)

    with lock:
        state = random.getstate()
        random.seed(hash(obj))
        for i in range(tensor_length):
            vec[i] = random.uniform(-1, 1)
        random.setstate(state)

    return vec


def print_tensor_stats(tensor):
    min = torch.min(tensor)
    max = torch.max(tensor)
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    median = torch.median(tensor)
    print('min: {}, max: {}, mean: {}, std: {}, median: {}'.format(min, max, mean, std, median))


def print_numpy_stats(numpy_array):
    min = np.min(numpy_array)
    max = np.max(numpy_array)
    mean = np.mean(numpy_array)
    std = np.std(numpy_array)
    median = np.median(numpy_array)
    print('min: {}, max: {}, mean: {}, std: {}, median: {}'.format(min, max, mean, std, median))
