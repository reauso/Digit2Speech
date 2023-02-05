import os
from data_handling.Dataset import DigitAudioDataset
import numpy as np

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'Dataset', 'training')
    dataset = DigitAudioDataset(data_path, audio_sample_coverage=0.2, shuffle_audio_samples=False)

    data_pair = dataset[0]
    print(len(dataset))
    print(data_pair[0])
    print(data_pair[1])
    print(data_pair[2])
    print(len(data_pair[1]))
    print(len(data_pair[2]))
    print(len(np.unique(data_pair[2])))