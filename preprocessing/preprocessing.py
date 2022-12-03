import os
import os.path as path

import librosa
import numpy as np
from scipy.signal import argrelmin, argrelmax

import util

if __name__ == '__main__':
    os.chdir(path.dirname(path.dirname(path.realpath(__file__))))
    raw_samples_directory = path.join(os.getcwd(), 'raw-samples')

    audio_files = util.files_in_directory(raw_samples_directory)
    print(audio_files)
    print(len(audio_files))

    threshold = -0.04

    for file in audio_files:
        # load file
        signal, sample_rate = librosa.load(file, sr=librosa.get_samplerate(file))

        local_minima_indices = argrelmin(signal)[0]
        local_minima = signal[local_minima_indices]
        relevant_local_minima_indices = np.where(local_minima < threshold)[0]

        #print(local_minima_indices)
        #print(local_minima)
        print('relevant_local_minima_indices: {}'.format(relevant_local_minima_indices))
        #print(local_minima[relevant_local_minima_indices])

        index_difference_of_relevant_local_minima = np.diff(relevant_local_minima_indices)
        relevant_index_differences_indices = np.where(index_difference_of_relevant_local_minima > 10000)[0]
        relevant_index_differences = index_difference_of_relevant_local_minima[relevant_index_differences_indices]
        #local_maxima_of_index_difference_indices = argrelmax(relevant_index_differences)[0]

        print('index_difference_of_relevant_local_minima: {}'.format(index_difference_of_relevant_local_minima))
        print('relevant_index_differences: {}'.format(relevant_index_differences_indices))
        print('relevant_index_differences: {}'.format(relevant_index_differences))
        #print('local_maxima_of_index_difference_indices: {}'.format(local_maxima_of_index_difference_indices))
        print('signal indices of relevant minima: {}'.format(relevant_local_minima_indices[relevant_index_differences_indices]))
        print(len(relevant_index_differences_indices))
        print('{}'.format(relevant_local_minima_indices[relevant_index_differences_indices] + (relevant_index_differences / 2)))
        print(signal)
        print(signal[np.floor(relevant_local_minima_indices[relevant_index_differences_indices] + (relevant_index_differences / 2))])
        #print('c: {}'.format(relevant_local_minima_indices[relevant_index_differences_indices[local_maxima_of_index_difference_indices]]))
        #print('a: {}'.format(relevant_local_minima_indices[index_difference_of_relevant_local_minima[local_maxima_of_index_difference_indices]]))
        #print('b: {}'.format(local_minima_indices[relevant_local_minima_indices[local_maxima_of_index_difference_indices]]))
        #print(local_minima_indices[local_maxima_of_index_difference_indices])
        exit()


