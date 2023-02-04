import glob
import os.path

import soundfile
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


def get_audio_file_path(folder, language, speaker, digit, trial):
    return os.path.join(folder, f"lang-{language}_speaker-{speaker}_digit-{digit}_trial-{trial}.wav")


def get_metadata_from_file_name(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split("_")
    language = parts[0].split("-")[1]
    speaker = parts[1].split("-")[1]
    digit = int(parts[2].split("-")[1])
    trial = parts[3].split("-")[1].split(".")[0]
    return language, speaker, digit, trial


def write_trial_to_file(output_folder, language, speaker, digit, trial, signal, sample_rate):
    soundfile.write(get_audio_file_path(output_folder, language,
                    speaker, digit, trial), signal, sample_rate)


def normalize_tensor(tensor, min_value=None, max_value=None):
    """
    Normalizes a tensor to range [-1;1]
    """
    min_value = min_value if min_value is not None else torch.min(tensor)
    max_value = max_value if max_value is not None else torch.max(tensor)
    delta = max_value - min_value

    if delta == 0:
        return tensor - min_value

    scale_factor = 2 / delta
    tensor = tensor * scale_factor
    return tensor - 1
