import glob
import io
import json
import os.path
import random
import sys
import threading
from datetime import datetime

import numpy as np
import soundfile
import re
import torch
from matplotlib import pyplot as plt


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


def read_textfile(textfile, mode='text', encoding='utf-8'):
    """
    Reads a textfile.
    :param textfile: The textfile path.
    :param mode: Determines the return type. 'lines' for a list of textfile lines or 'text' for one string containing
    all file content.
    :param encoding: The encoding of the textfile.
    :return: The content of the textfile.
    """
    f = open(textfile, 'r', encoding=encoding)
    if mode == 'lines':
        text = f.readlines()
    elif mode == 'text':
        text = f.read()
    else:
        raise NotImplementedError('The given mode {} is not implemented!'.format(mode))
    f.close()

    return text


def read_jsonfile(path) -> dict:
    f = open(path, 'r')
    obj = json.load(f)
    f.close()
    return obj


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


def signal_to_image(signal, dpi=100):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(signal)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='raw', dpi=dpi)
    signal_img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    signal_img = signal_img.reshape((int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    plt.close(fig)

    return signal_img


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
    size = tensor.size()
    print('min: {}, max: {}, mean: {}, std: {}, median: {}, size: {}'.format(min, max, mean, std, median, size))


def print_numpy_stats(numpy_array):
    min = np.min(numpy_array)
    max = np.max(numpy_array)
    mean = np.mean(numpy_array)
    std = np.std(numpy_array)
    median = np.median(numpy_array)
    shape = numpy_array.shape()
    print('min: {}, max: {}, mean: {}, std: {}, median: {}, shape: {}'.format(min, max, mean, std, median, shape))


def latest_experiment_path(checkpoint_dir):
    checkpoint_files = files_in_directory(checkpoint_dir)
    checkpoint_files = [file for file in checkpoint_files if os.path.isdir(file)]

    date_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    date_matches = [re.search(date_pattern, name) for name in checkpoint_files]

    date_path_tuples = zip(date_matches, checkpoint_files)
    date_path_tuples = [(date.group(0), path) for date, path in date_path_tuples if date is not None]
    date_path_tuples = [(datetime.strptime(date, '%Y-%m-%d_%H-%M-%S'), path) for date, path in date_path_tuples]

    date_path_tuples.sort(key=lambda x: (x[0], x[1]), reverse=True)

    latest_experiment_path = date_path_tuples[0][1]

    return latest_experiment_path


def all_trial_paths(experiment_dir, trial_name_pattern=r'train_.{5}_\d{5}'):
    trial_file_names = ['checkpoint', 'params.json', 'params.pkl', 'progress.csv', 'result.json']

    trial_paths = files_in_directory(experiment_dir)
    trial_paths = [path for path in trial_paths if os.path.isdir(path)]
    trial_paths = [path for path in trial_paths if re.search(trial_name_pattern, path) is not None]

    # validation of files
    for name in trial_file_names:
        trial_paths = [path for path in trial_paths if os.path.exists(os.path.join(path, name))]

    return trial_paths


def best_trial_path(experiment_dir, metric='eval_loss', lower_is_better=True, minimal_iterations='highest'):
    trial_paths = all_trial_paths(experiment_dir=experiment_dir)
    best_path = None
    best_value = sys.maxsize if lower_is_better else -sys.maxsize
    all_results = []

    # get all results
    for path in trial_paths:
        result = read_textfile(os.path.join(path, 'result.json'), mode='lines')[-2]
        result = json.loads(result)
        all_results.append(result)

    # if automatic iteration detection get the highest training_iteration
    if minimal_iterations == 'highest':
        minimal_iterations = max([result['training_iteration'] for result in all_results])

    # filter for minimal_iterations
    path_result_pairs = zip(trial_paths, all_results)
    path_result_pairs = [pair for pair in path_result_pairs if pair[1]['training_iteration'] >= minimal_iterations]

    # check results of all trial paths
    for path, result in path_result_pairs:
        value = result[metric]

        if lower_is_better and best_value > value:
            best_path = path
            best_value = value
        elif not lower_is_better and best_value < value:
            best_path = path
            best_value = value

    return best_path
