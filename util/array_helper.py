import io
import random
import threading

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')


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


def signal_to_image(signal, color='tab:blue', dpi=100, lines_at=None, colors=None):
    lines_at = lines_at if lines_at is not None else []
    colors = colors if colors is not None else ['red' for _ in lines_at]
    if len(colors) < len(lines_at):
        colors.extend(['red' for _ in range(len(lines_at) - len(colors))])

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(signal, color=color)
    for i, line in enumerate(lines_at):
        ax.axvline(x=line, color=colors[i])
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
    shape = numpy_array.shape
    print('min: {}, max: {}, mean: {}, std: {}, median: {}, shape: {}'.format(min, max, mean, std, median, shape))
