from __future__ import annotations

import json
import os
from typing import List

from util.checkpoint_helper import latest_experiment_path, all_trial_paths, trial_short_name
from util.data_helper import read_jsonfile, read_textfile


class RayTuneAnalysis:
    def __init__(self, checkpoint_directory, experiment_names: str | List[str] = 'latest_only',
                 excluded_fields=None):
        # set parameters
        self._checkpoint_directory = checkpoint_directory
        self._excluded_result_fields = [] if excluded_fields is None else excluded_fields

        # define necessary variables
        self._experiment_paths = None
        self._trial_paths = None
        self._statistics = {}
        self._config_analysis = {}
        self.trial_file_names = ['checkpoint', 'params.json', 'params.pkl', 'progress.csv', 'result.json']

        # determine paths
        self._determine_experiment_paths(experiment_names)
        self._determine_all_trial_paths()

        # add each trial to statistics
        self._read_and_add_trial_statistics()

        # config analysis
        self.determine_config_analysis()

    def _determine_experiment_paths(self, experiment_names):
        if experiment_names == 'latest_only':
            self._experiment_paths = [latest_experiment_path(self._checkpoint_directory)]
        else:
            self._experiment_paths = [os.path.join(self._checkpoint_directory, name) for name in experiment_names]
            self._experiment_paths = [path for path in self._experiment_paths if os.path.isdir(path)]

        print("Analyse experiments with names '{}'".format([os.path.basename(path) for path in self._experiment_paths]))

    def _determine_all_trial_paths(self):
        self._trial_paths = []
        for experiment_path in self._experiment_paths:
            self._trial_paths.extend(all_trial_paths(experiment_dir=experiment_path))

    def _read_and_add_trial_statistics(self):
        for trial_path in self._trial_paths:
            trial_name = trial_short_name(trial_path)

            # read config and results
            config = read_jsonfile(os.path.join(trial_path, 'params.json'))
            result = read_textfile(os.path.join(trial_path, 'result.json'), mode='lines')[-2]
            result = json.loads(result)

            # exclude fields
            for field in self._excluded_result_fields:
                if field in config:
                    del config[field]
                if field in result:
                    del result[field]

            # add to statistics
            self._statistics[trial_name] = {'config': config, 'result': result}

    def determine_config_analysis(self):
        # add all config fields to config analysis
        for trial in self._statistics.values():
            config_fields = trial['config'].keys()
            self._config_analysis.update({key: {} for key in config_fields})

        # add all config field values to config analysis
        for trial in self._statistics.values():
            for config_field, config_value in trial['config'].items():
                self._config_analysis[config_field][config_value] = {}

        # add result fields to each config field value
        for trial in self._statistics.values():
            for config_field, config_value in trial['config'].items():
                empty_result_key_dict = {result_field: [] for result_field in trial['result'].keys()}
                self._config_analysis[config_field][config_value].update(empty_result_key_dict)

        # add each trial result values to the config field values
        for trial in self._statistics.values():
            for config_field, config_value in trial['config'].items():
                for result_field, result_value in trial['result'].items():
                    self._config_analysis[config_field][config_value][result_field].append(result_value)

    @property
    def experiment_paths(self):
        return self._experiment_paths

    @property
    def statistics(self):
        return self._statistics

    @property
    def config_analysis(self):
        return self._config_analysis
