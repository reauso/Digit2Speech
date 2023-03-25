from __future__ import annotations

import json
import math
import os
from enum import Enum
from typing import List

import numpy as np

from util.checkpoint_helper import latest_experiment_path, trial_short_name, all_trial_paths, best_trial_path
from util.data_helper import files_in_directory, read_textfile, read_jsonfile, write_textfile


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
            self._experiment_paths = [latest_experiment_path(checkpoint_dir)]
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


class RayTuneAnalysisTableView:
    class _TableType(Enum):
        statistics = 'statistics'
        config_analysis = 'config_analysis'

    def __init__(self, analysis: RayTuneAnalysis):
        self._analysis = analysis

        self._statistics_table = ''
        self._config_analysis_table = ''
        self._column_key_order = []
        self._column_widths = {}

        # calculate tables
        self._determine_statistics_table()
        self._determine_config_analysis_table()

    def _determine_statistics_table(self):
        self._column_key_order = []
        self._column_widths = {}
        self._determine_statistics_order_and_widths()

        # construct table
        self._append_dividing_line(self._TableType.statistics)
        self._append_empty_line(self._TableType.statistics)
        self._append_header_line(self._TableType.statistics)
        self._append_empty_line(self._TableType.statistics)
        self._append_dividing_line(self._TableType.statistics)

        for key, value in self._analysis.statistics.items():
            self._append_statistics_row(key, value['config'], value['result'])
        self._append_dividing_line(self._TableType.statistics)

    def _determine_statistics_order_and_widths(self):
        key_max_width, config_max_widths, result_max_widths = self._determine_statistics_column_widths()

        # order
        self._column_key_order.append('Trial')
        self._column_key_order.extend(list(config_max_widths.keys()))
        self._column_key_order.extend(list(result_max_widths.keys()))

        # widths
        self._column_widths['Trial'] = key_max_width
        self._column_widths.update(config_max_widths)
        self._column_widths.update(result_max_widths)

    def _determine_statistics_column_widths(self):
        key_max_width = len('Trial')
        config_max_widths = {}
        result_max_widths = {}

        # key max length
        for key in self._analysis.statistics.keys():
            key_max_width = max(key_max_width, len(key))

        # config init
        for trial_value in self._analysis.statistics.values():
            config = trial_value['config']

            for key in config.keys():
                config_max_widths[key] = len(key)

        # result init
        for trial_value in self._analysis.statistics.values():
            result = trial_value['result']

            for key in result.keys():
                result_max_widths[key] = len(key)

        # config max length
        for trial_value in self._analysis.statistics.values():
            config = trial_value['config']

            for key, value in config.items():
                config_max_widths[key] = max(config_max_widths[key], len(str(value)))

        # result max length
        for trial_value in self._analysis.statistics.values():
            result = trial_value['result']

            for key, value in result.items():
                result_max_widths[key] = max(result_max_widths[key], len(str(value)))

        # add padding and divider
        key_max_width += 3
        config_max_widths = {key: value + 3 for key, value in config_max_widths.items()}
        result_max_widths = {key: value + 3 for key, value in result_max_widths.items()}

        return key_max_width, config_max_widths, result_max_widths

    def _append_statistics_row(self, trial_name, config, result):
        values_dict = {'Trial': trial_name}
        values_dict.update(config)
        values_dict.update(result)

        for key in self._column_key_order:
            self._append_field(key, values_dict[key], self._TableType.statistics, center=False)

        self._statistics_table += '|\n'

    def _determine_config_analysis_table(self):
        self._column_key_order = []
        self._column_widths = {}
        self._determine_config_analysis_order_and_widths()

        # construct table
        self._append_dividing_line(self._TableType.config_analysis)
        self._append_empty_line(self._TableType.config_analysis)
        self._append_header_line(self._TableType.config_analysis)
        self._append_empty_line(self._TableType.config_analysis)
        self._append_dividing_line(self._TableType.config_analysis)
        self._append_dividing_line(self._TableType.config_analysis)

        for config_field, config_values in self._analysis.config_analysis.items():
            self._append_config_analysis_config_block(config_field, config_values)
        self._append_dividing_line(self._TableType.statistics)

    def _determine_config_analysis_order_and_widths(self):
        config_field_width, config_value_width, result_max_widths = self._determine_config_analysis_column_widths()

        # order
        self._column_key_order.append('Config Field')
        self._column_key_order.append('Config Value')
        self._column_key_order.extend(list(result_max_widths.keys()))

        # widths
        self._column_widths['Config Field'] = config_field_width
        self._column_widths['Config Value'] = config_value_width
        self._column_widths.update(result_max_widths)

    def _determine_config_analysis_column_widths(self):
        config_field_width = len('Config Field')
        config_value_width = len('Config Value')
        result_max_widths = {}

        # determine config field width
        for config_field in self._analysis.config_analysis.keys():
            config_field_width = max(config_field_width, len(config_field))

        # determine config value width
        for config_field, config_values in self._analysis.config_analysis.items():
            for key in config_values.keys():
                config_value_width = max(config_value_width, len(str(key)))

        # init result_max_widths
        for config_field, config_values in self._analysis.config_analysis.items():
            for config_value, result in config_values.items():
                for key in result.keys():
                    result_max_widths[key] = len(key)

        # determine config value results width
        for config_field, config_values in self._analysis.config_analysis.items():
            for config_value, result in config_values.items():
                for key, value in result.items():
                    result_max_widths[key] = max(result_max_widths[key], len('{:.5f}'.format(np.mean(np.array(value)))))

        # append padding
        config_field_width += 3
        config_value_width += 3
        result_max_widths = {key: value + 3 for key, value in result_max_widths.items()}

        return config_field_width, config_value_width, result_max_widths

    def _append_config_analysis_config_block(self, config_field, config_values):
        num_config_values = len(config_values)
        config_field_row = math.floor(num_config_values / 2)

        for row, (config_value, result) in enumerate(config_values.items()):
            values_dict = {'Config Field': config_field} if row == config_field_row else {'Config Field': ''}
            values_dict['Config Value'] = config_value
            values_dict.update({key: '{:.5f}'.format(np.mean(np.array(value))) for key, value in result.items()})

            for key in self._column_key_order:
                self._append_field(key, values_dict[key], self._TableType.config_analysis, center=False)

            self._config_analysis_table += '|\n'

        self._append_dividing_line(self._TableType.config_analysis)

    def _append_dividing_line(self, table_type: _TableType):
        line = ''

        for key in self._column_key_order:
            line += '+'
            line += ''.join('-' for _ in range(self._column_widths[key] - 1))

        line += '+\n'

        if table_type is self._TableType.statistics:
            self._statistics_table += line
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_table += line

    def _append_empty_line(self, table_type: _TableType):
        line = ''

        for key in self._column_key_order:
            line += '|'
            line += ''.join(' ' for _ in range(self._column_widths[key] - 1))

        line += '|\n'

        if table_type is self._TableType.statistics:
            self._statistics_table += line
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_table += line

    def _append_header_line(self, table_type: _TableType):
        for key in self._column_key_order:
            self._append_field(key, key, table_type)

        if table_type is self._TableType.statistics:
            self._statistics_table += '|\n'
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_table += '|\n'

    def _append_field(self, key, value, table_type: _TableType, center=True):
        field_width = self._column_widths[key]
        value_space = field_width - 3
        value_length = len(str(value))
        padding = math.floor((value_space - value_length) / 2)

        text = '| '
        if center:
            text += ''.join([' ' for _ in range(padding)])
        text += str(value)
        text += ''.join([' ' for _ in range(field_width - len(text))])

        if table_type is self._TableType.statistics:
            self._statistics_table += text
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_table += text

    @property
    def statistics_table(self):
        return self._statistics_table

    @property
    def config_analysis_table(self):
        return self._config_analysis_table


if __name__ == "__main__":
    # config
    checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    experiment_names = []
    experiment_names = 'latest_only'
    default_excluded_fields = ['eval_vid', 'train_vid', 'time_this_iter_s', 'done', 'timesteps_total',
                               'episodes_total', 'trial_id', 'experiment_id', 'date', 'timestamp', 'pid', 'hostname',
                               'node_ip', 'config', 'time_since_restore', 'timesteps_since_restore',
                               'iterations_since_restore', 'warmup_time', 'transformation_file',
                               'training_dataset_path', 'validation_dataset_path', 'feature_mapping_file']

    analysis = RayTuneAnalysis(checkpoint_dir, experiment_names=experiment_names,
                               excluded_fields=default_excluded_fields)
    table_view = RayTuneAnalysisTableView(analysis)
    print(table_view.statistics_table)
    print(table_view.config_analysis_table)

    # save to file if it is a single experiment
    experiment_paths = analysis.experiment_paths
    if isinstance(experiment_paths, list) and len(experiment_paths) == 1:
        experiment_path = experiment_paths[0]
        text = table_view.statistics_table + '\n\n\n' + table_view.config_analysis_table + '\n\n\n'
        text += 'Best Trial Path: ' + best_trial_path(experiment_path)

        analysis_file = os.path.join(experiment_path, 'analysis.txt')
        write_textfile(text, analysis_file)
