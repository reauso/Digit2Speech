import json
import os
import re
import sys
from datetime import datetime

from util.data_helper import files_in_directory, read_textfile


def experiment_datetime(experiment_name):
    date_pattern = r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
    date_match = re.search(date_pattern, experiment_name)
    date_match = date_match.group(0) if date_match is not None else None
    date = datetime.strptime(date_match, '%Y-%m-%d_%H-%M-%S') if date_match is not None else None

    return date


def latest_experiment_path(checkpoint_dir):
    checkpoint_files = files_in_directory(checkpoint_dir)
    checkpoint_files = [file for file in checkpoint_files if os.path.isdir(file)]
    dates = [experiment_datetime(os.path.basename(path)) for path in checkpoint_files]

    date_path_tuples = zip(dates, checkpoint_files)
    date_path_tuples = [(date, path) for date, path in date_path_tuples if date is not None]
    date_path_tuples.sort(key=lambda x: (x[0], x[1]), reverse=True)

    latest_experiment_path = date_path_tuples[0][1]

    return latest_experiment_path


def nearest_experiment_path(checkpoint_dir, datetime):
    checkpoint_files = files_in_directory(checkpoint_dir)
    checkpoint_files = [file for file in checkpoint_files if os.path.isdir(file)]
    dates = [experiment_datetime(os.path.basename(path)) for path in checkpoint_files]

    date_path_tuples = zip(dates, checkpoint_files)
    date_path_tuples = [(date, path) for date, path in date_path_tuples if date is not None]

    nearest_experiment_path = min(date_path_tuples, key=lambda x: abs(x[0] - datetime))[1]

    return nearest_experiment_path


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


def trial_short_name(full_name, short_name_pattern=r'train_(.{5}_\d{5})_\d+_'):
    return re.search(short_name_pattern, os.path.basename(full_name)).group(1)
