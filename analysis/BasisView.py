from abc import ABC, abstractmethod

from analysis.RayTuneAnalysis import RayTuneAnalysis


class BasisView(ABC):
    def __init__(self, analysis: RayTuneAnalysis):
        self._analysis = analysis

        self._statistics_view = None
        self._config_analysis_view = None

        # init views
        self._init_views()

        # calculate views
        self._determine_statistics_view()
        self._determine_config_analysis_view()

    @abstractmethod
    def _init_views(self):
        pass

    @abstractmethod
    def _determine_statistics_view(self):
        pass

    @abstractmethod
    def _determine_config_analysis_view(self):
        pass

    @property
    def statistics_view(self):
        return self._statistics_view

    @property
    def config_analysis_view(self):
        return self._config_analysis_view
