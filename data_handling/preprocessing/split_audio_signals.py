import os

import librosa
import numpy as np
from scipy.signal import argrelmin
from tqdm import tqdm

from data_handling import util


def trial_midpoint_indices_in_signal(signal, sample_rate, threshold, minimal_trial_sample_size):
    # get local minima with intensities below threshold
    local_minima_indices = argrelmin(signal)[0]
    local_minima = signal[local_minima_indices]
    relevant_local_minima_indices = np.where(local_minima < -abs(threshold))[0]
    relevant_local_minima_signal_indices = local_minima_indices[relevant_local_minima_indices]

    # now get the minima where the next minimum is over 10000 samples away
    index_difference_of_relevant_local_minima = np.diff(relevant_local_minima_indices)
    relevant_index_differences_indices = np.where(index_difference_of_relevant_local_minima > 10000)[0]

    # create a 2-dimensional array containing the beginning and ending index of each trial
    trials_indices = np.concatenate(
        [
            relevant_index_differences_indices,
            relevant_index_differences_indices + 1,
            [0, -1]
        ],
        axis=0)
    trials_signal_indices = relevant_local_minima_signal_indices[trials_indices]
    trials_signal_indices = np.sort(trials_signal_indices, kind='quicksort')
    number_of_trials = relevant_index_differences_indices.shape[0] + 1
    trials_signal_indices = np.reshape(trials_signal_indices, (number_of_trials, 2))

    # sort out trials which sample length is below minimal_trial_sample_size
    relevant_trials_indices = np.where(np.diff(trials_signal_indices, axis=1) > minimal_trial_sample_size)[0]
    relevant_trials_signal_indices = trials_signal_indices[relevant_trials_indices]
    number_of_trials = relevant_trials_signal_indices.shape[0]

    # calculate the midpoint index of each remaining trial in the signal array
    trial_mid_signal_indices = relevant_trials_signal_indices[:, 0] + np.reshape(np.floor(
        np.diff(relevant_trials_signal_indices, axis=1) / 2), number_of_trials).astype(int)

    # remove trials which cannot be cropped to 2 seconds because of missing padding data
    trial_mid_signal_indices = trial_mid_signal_indices[np.intersect1d(
        np.where(trial_mid_signal_indices >= sample_rate)[0],
        np.where(trial_mid_signal_indices < signal.shape[0] - sample_rate)[0],
    )]

    return trial_mid_signal_indices