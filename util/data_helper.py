import glob
import json
import os
import re

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


def write_textfile(text, textfile, encoding='utf-8'):
    f = open(textfile, 'w', encoding=encoding)
    f.write(text)
    f.close()


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
