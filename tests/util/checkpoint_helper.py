import datetime
from unittest import TestCase
from unittest.mock import patch

from parameterized import parameterized

from util.checkpoint_helper import latest_experiment_path, experiment_datetime, nearest_experiment_path


class CheckpointHelperTest(TestCase):

    @parameterized.expand([
        ['train_2022-07-09_00-36-10', datetime.datetime(year=2022, month=7, day=9, hour=0, minute=36, second=10)],
        ['train_2023-03-13_12-20-23', datetime.datetime(year=2023, month=3, day=13, hour=12, minute=20, second=23)],
        ['train_2023-02-11_20-49-09', datetime.datetime(year=2023, month=2, day=11, hour=20, minute=49, second=9)],
        ['C:\\train_2023-03-06_21-44-07', datetime.datetime(year=2023, month=3, day=6, hour=21, minute=44, second=7)],
    ])
    def test_experiment_datetime__valid_date__returns_correct_datetime(self, experiment_name, real_date):
        date = experiment_datetime(experiment_name)

        message = 'Params: experiment_name: {}, real_date: {}'.format(experiment_name, real_date)
        self.assertIsNotNone(date, msg=message)
        self.assertEqual(date, real_date, msg=message)

    @parameterized.expand([
        ['train_07-09_2022_00-36-10', datetime.datetime(year=2022, month=7, day=9, hour=0, minute=36, second=10)],
        ['jiop', datetime.datetime(year=2023, month=3, day=13, hour=12, minute=20, second=23)],
        ['', datetime.datetime(year=2023, month=2, day=11, hour=20, minute=49, second=9)],
        ['test_train_21-44-07_2023-03-06', datetime.datetime(year=2023, month=3, day=6, hour=21, minute=44, second=7)],
    ])
    def test_experiment_datetime__invalid_date__returns_none(self, experiment_name, real_date):
        date = experiment_datetime(experiment_name)

        message = 'Params: experiment_name: {}, real_date: {}'.format(experiment_name, real_date)
        self.assertIsNone(date, msg=message)
        self.assertNotEqual(date, real_date, msg=message)

    @patch('util.checkpoint_helper.os.path.isdir')
    @patch('util.checkpoint_helper.files_in_directory')
    def test_latest_experiment_path__valid_files__returns_path_of_latest_experiment(self, files_in_directory_mock,
                                                                                    os_isdir_mock):
        experiments = [
            'C:\\train_2022-07-09_00-36-10',
            'C:\\train_2023-03-13_12-20-23',
            'C:\\train_2023-02-11_20-49-09',
            'C:\\train_2023-03-06_21-44-07',
        ]
        files_in_directory_mock.return_value = experiments
        os_isdir_mock.return_value = True

        path = latest_experiment_path('C:\\')

        self.assertEqual(path, 'C:\\train_2023-03-13_12-20-23')

    @patch('util.checkpoint_helper.os.path.isdir')
    @patch('util.checkpoint_helper.files_in_directory')
    def test_nearest_experiment_path__valid_files_returns_path_of_nearest_experiment(self, files_in_directory_mock,
                                                                                     os_isdir_mock):
        experiments = [
            'C:\\train_2022-07-09_00-36-10',
            'C:\\train_2023-03-13_12-20-23',
            'C:\\train_2023-02-11_20-49-09',
            'C:\\train_2023-03-06_21-44-07',
        ]

        files_in_directory_mock.return_value = experiments
        os_isdir_mock.return_value = True

        date = datetime.datetime(year=2023, month=3, day=7, hour=10, minute=20, second=56)
        path = nearest_experiment_path('C:\\', date)

        self.assertEqual(path, 'C:\\train_2023-03-06_21-44-07')

