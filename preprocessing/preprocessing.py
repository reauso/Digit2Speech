import os
import os.path as path

import librosa

import util

if __name__ == '__main__':
    os.chdir(path.dirname(path.dirname(path.realpath(__file__))))
    raw_samples_directory = path.join(os.getcwd(), 'raw-samples')

    audio_files = util.files_in_directory(raw_samples_directory)
    print(audio_files)
    print(len(audio_files))

    for file in audio_files:
        # load file
        signal, sample_rate = librosa.load(file, sr=librosa.get_samplerate(file))


