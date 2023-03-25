from __future__ import annotations

import os

from analysis.RayTuneAnalysis import RayTuneAnalysis
from analysis.TextTableView import TextTableView
from util.checkpoint_helper import best_trial_path
from util.data_helper import write_textfile

if __name__ == "__main__":
    # config
    view_class = TextTableView
    # view_class = MarkupTableView
    checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    experiment_names = 'latest_only'
    default_excluded_fields = ['eval_vid', 'train_vid', 'time_this_iter_s', 'done', 'timesteps_total',
                               'episodes_total', 'trial_id', 'experiment_id', 'date', 'timestamp', 'pid', 'hostname',
                               'node_ip', 'config', 'time_since_restore', 'timesteps_since_restore',
                               'iterations_since_restore', 'warmup_time', 'transformation_file',
                               'training_dataset_path', 'validation_dataset_path', 'feature_mapping_file']

    # analysis and view
    analysis = RayTuneAnalysis(checkpoint_dir, experiment_names=experiment_names,
                               excluded_fields=default_excluded_fields)
    table_view = view_class(analysis)
    print(table_view.statistics_view)
    print(table_view.config_analysis_view)

    # save to file if it is a single experiment
    experiment_paths = analysis.experiment_paths
    if isinstance(experiment_paths, list) and len(experiment_paths) == 1:
        experiment_path = experiment_paths[0]
        text = table_view.statistics_view + '\n\n\n' + table_view.config_analysis_view + '\n\n\n'
        text += 'Best Trial Path: ' + best_trial_path(experiment_path)

        analysis_file = os.path.join(experiment_path, 'analysis.txt')
        write_textfile(text, analysis_file)
