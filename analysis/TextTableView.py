import math
from enum import Enum

import numpy as np

from analysis.BasisView import BasisView
from analysis.RayTuneAnalysis import RayTuneAnalysis


class TextTableView(BasisView):
    class _TableType(Enum):
        statistics = 'statistics'
        config_analysis = 'config_analysis'

    def __init__(self, analysis: RayTuneAnalysis):
        self._column_key_order = []
        self._column_widths = {}

        super().__init__(analysis)

    def _init_views(self):
        self._statistics_view = ''
        self._config_analysis_view = ''

    def _determine_statistics_view(self):
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

        self._statistics_view += '|\n'

    def _determine_config_analysis_view(self):
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

            self._config_analysis_view += '|\n'

        self._append_dividing_line(self._TableType.config_analysis)

    def _append_dividing_line(self, table_type: _TableType):
        line = ''

        for key in self._column_key_order:
            line += '+'
            line += ''.join('-' for _ in range(self._column_widths[key] - 1))

        line += '+\n'

        if table_type is self._TableType.statistics:
            self._statistics_view += line
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_view += line

    def _append_empty_line(self, table_type: _TableType):
        line = ''

        for key in self._column_key_order:
            line += '|'
            line += ''.join(' ' for _ in range(self._column_widths[key] - 1))

        line += '|\n'

        if table_type is self._TableType.statistics:
            self._statistics_view += line
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_view += line

    def _append_header_line(self, table_type: _TableType):
        for key in self._column_key_order:
            self._append_field(key, key, table_type)

        if table_type is self._TableType.statistics:
            self._statistics_view += '|\n'
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_view += '|\n'

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
            self._statistics_view += text
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_view += text
