from enum import Enum

import numpy as np

from analysis.BasisView import BasisView
from analysis.RayTuneAnalysis import RayTuneAnalysis


class MarkupTableView(BasisView):
    class _TableType(Enum):
        statistics = 'statistics'
        config_analysis = 'config_analysis'

    def __init__(self, analysis: RayTuneAnalysis):
        self._column_key_order = []

        super().__init__(analysis)

    def _init_views(self):
        self._statistics_view = ''
        self._config_analysis_view = ''

    def _determine_statistics_view(self):
        # determine order
        self._determine_statistics_order()

        # construct table
        self._append_header_line(self._TableType.statistics)
        self._append_dividing_line(self._TableType.statistics)

        for key, value in self._analysis.statistics.items():
            self._append_statistics_row(key, value['config'], value['result'])

    def _determine_statistics_order(self):
        # get all config and result keys
        config_keys = []
        result_keys = []

        for trial, value in self._analysis.statistics.items():
            config_keys.extend(list(value['config'].keys()))
            result_keys.extend(list(value['result'].keys()))

            config_keys = list(set(config_keys))
            result_keys = list(set(result_keys))

        # order
        self._column_key_order.append('Trial')
        self._column_key_order.extend(config_keys)
        self._column_key_order.extend(result_keys)

    def _append_statistics_row(self, trial_name, config, result):
        # result values clipping
        result = {key: '{:.5f}'.format(value) for key, value in result.items()}

        values_dict = {'Trial': trial_name}
        values_dict.update(config)
        values_dict.update(result)

        for key in self._column_key_order:
            self._append_field(key, values_dict[key], self._TableType.statistics)

        self._statistics_view += '|\n'

    def _determine_config_analysis_view(self):
        # reset
        self._column_key_order = []

        # determine order
        self._determine_config_analysis_order()

        # construct table
        self._append_header_line(self._TableType.config_analysis)
        self._append_dividing_line(self._TableType.config_analysis)

        for config_field, config_values in self._analysis.config_analysis.items():
            self._append_config_analysis_config_block(config_field, config_values)

    def _determine_config_analysis_order(self):
        # get all config and result keys
        result_keys = []
        for _, config_values in self._analysis.config_analysis.items():
            for _, result in config_values.items():
                result_keys.extend(result.keys())
                result_keys = list(set(result_keys))
        # order
        self._column_key_order.append('Config Field')
        self._column_key_order.append('Config Value')
        self._column_key_order.extend(result_keys)

    def _append_config_analysis_config_block(self, config_field, config_values):
        for row, (config_value, result) in enumerate(config_values.items()):
            values_dict = {'Config Field': config_field, 'Config Value': config_value}
            values_dict.update({key: '{:.5f}'.format(np.mean(np.array(value))) for key, value in result.items()})

            for key in self._column_key_order:
                self._append_field(key, values_dict[key], self._TableType.config_analysis)

            self._config_analysis_view += '|\n'

    def _append_header_line(self, table_type: _TableType):
        line = ''.join('| {} '.format(key) for key in self._column_key_order)
        line += '|\n'

        self.add_text_to_table(line, table_type)

    def _append_dividing_line(self, table_type: _TableType):
        line = ''.join('| - ' for _ in self._column_key_order)
        line += '|\n'

        self.add_text_to_table(line, table_type)

    def _append_field(self, key, value, table_type: _TableType):
        text = '| ' + str(value) + ' '

        self.add_text_to_table(text, table_type)

    def add_text_to_table(self, text, table_type):
        if table_type is self._TableType.statistics:
            self._statistics_view += text
        elif table_type is self._TableType.config_analysis:
            self._config_analysis_view += text
