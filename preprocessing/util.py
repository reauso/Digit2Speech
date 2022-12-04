import glob
import os.path

import soundfile


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


def get_parts_from_file_path(file_path):
    file_name = os.path.basename(file_path)
    parts = file_name.split("_")
    language = parts[0].split("-")[1]
    speaker = parts[1].split("-")[1]
    digit = parts[2].split("-")[1]
    trial = parts[3].split("-")[1].split(".")[0]
    return language, speaker, digit, trial


def write_trial_to_file(output_folder, language, speaker, digit, trial, signal, sample_rate):
    soundfile.write(get_audio_file_path(output_folder, language,
                    speaker, digit, trial), signal, sample_rate)
