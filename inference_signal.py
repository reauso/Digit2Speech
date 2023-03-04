import json
import os

import librosa
import numpy as np
import soundfile
import torch
from tqdm import tqdm

from data_handling.util import read_textfile, files_in_directory, get_metadata_from_file_name, normalize_tensor
from model.SirenModel import SirenModelWithFiLM

if __name__ == "__main__":
    # define values
    sample_rate = 48000
    seconds = 2
    use_metadata_from_file = True
    language = 'german'
    sex = 'male'
    digit = '2'

    # define necessary paths
    source_path = os.path.join(os.getcwd(), os.path.normpath('Dataset/validation'))
    save_path = os.path.join(os.getcwd(), 'GeneratedAudio')
    experiment_name = 'train_2023-03-04_17-30-15'
    trial_name = 'train_d8263_00000_0_2023-03-04_17-34-36'
    model_path = os.path.join(os.getcwd(), 'Checkpoints', experiment_name, trial_name)
    feature_mapping_file = os.path.normpath(os.getcwd() + '/data_handling/feature_mapping.json')
    transformation_file = os.path.normpath(os.getcwd() + '/Dataset/transformation.json')

    # create save dir
    os.makedirs(os.path.join(save_path, experiment_name), exist_ok=True)

    # load feature mappings
    f = open(feature_mapping_file)
    feature_mappings = json.load(f)
    speaker_sex_mapping = feature_mappings['speakers-sex']
    f.close()

    # load transformation
    if transformation_file is not None:
        raw_transformation_text = read_textfile(transformation_file)
        transformation = json.loads(raw_transformation_text)
        shift = transformation['shift']
        scale = 1 / transformation['scale']
    else:
        shift = 0.0
        scale = 1.0

    # create model instance
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_config = json.loads(read_textfile(os.path.join(model_path, 'params.json')))
    model = SirenModelWithFiLM(in_features=1,
                               out_features=1,
                               hidden_features=model_config["SIREN_hidden_features"],
                               num_layers=model_config["SIREN_num_layers"],
                               mod_features=model_config['num_mfccs'] + 3)
    model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model.to(device)

    # get all mfcc files from source path
    mfcc_files = files_in_directory(source_path, ['**/*_mfcc_{}.npy'.format(model_config['num_mfccs'])], recursive=True)
    print('Found {} MFCC Source files'.format(len(mfcc_files)))

    # load model state
    load_file_path = os.path.join(model_path, "checkpoint")
    print("Load from Checkpoint: {}".format(load_file_path))
    model_state, _ = torch.load(load_file_path)
    model.load_state_dict(model_state)

    # generate audio
    sample_indices = torch.arange(start=0.0, end=sample_rate * seconds, step=1, device=device, requires_grad=False)
    sample_indices = sample_indices[:, None]
    pipeline = tqdm(mfcc_files, unit='Files', desc='Generate Audio Files')
    for i, file in enumerate(pipeline):
        filename = os.path.basename(file)
        pipeline.postfix = filename

        # load mfcc coefficients
        mfcc_coefficients = np.load(file)
        mfcc_coefficients = mfcc_coefficients.reshape(mfcc_coefficients.shape[1])
        mfcc_coefficients = torch.FloatTensor(mfcc_coefficients)

        # load metadata from file if config is set
        if use_metadata_from_file:
            basename = os.path.splitext(filename)[0]
            language, speaker, digit, _ = get_metadata_from_file_name(basename)
            sex = speaker_sex_mapping[speaker]

        # define metadata dict
        metadata = {
            "language": language,
            "digit": digit,
            "sex": sex,
            "mfcc_coefficients": mfcc_coefficients,
        }

        # metadata to normalized float tensor
        num_mfcc = len(mfcc_coefficients)
        language = object_to_float_tensor(language, num_mfcc)
        sex = object_to_float_tensor(sex, num_mfcc)
        digit = object_to_float_tensor(digit, num_mfcc)

        # metadata to tensor
        modulation_input = torch.cat([
            metadata['language'],
            metadata['digit'],
            metadata['sex'],
            metadata['mfcc_coefficients']], dim=0).to(device, non_blocking=True)
        modulation_input = modulation_input[None, :]

        with torch.no_grad():
            signal = model(sample_indices, modulation_input)
            signal = signal.cpu().detach().numpy().reshape(signal.shape[0])
            signal = (signal * scale) + shift

            audio_file = os.path.join(source_path, 'lang-english_speaker-13_digit-0_trial-12.wav')
            original, _ = librosa.load(audio_file, sr=librosa.get_samplerate(audio_file), mono=True)
            print(original)
            print(signal)
            print('original max {}, min: {}, mean: {}, std: {}'.format(np.max(original), np.min(original), np.mean(original), np.std(original)))
            #print('max {}, min: {}, mean: {}, std: {}'.format(np.max(original2), np.min(original2), np.mean(original2), np.std(original2)))
            print('max {}, min: {}, mean: {}, std: {}'.format(np.max(signal), np.min(signal), np.mean(signal), np.std(signal)))

            filename = '{}-lang-{}-sex-{}-digit-{}.wav'.format(i, language, sex, digit)
            #soundfile.write(os.path.join(save_path, filename + 'orig.wav'), original, sample_rate)
            soundfile.write(os.path.join(save_path, experiment_name, filename), signal, sample_rate)
            exit()

            """
            audio_samples, audio_sampling_rate = librosa.load(file, sr=librosa.get_samplerate(file), mono=True)
            img = cv2.imread(spectrogram_file, cv2.IMREAD_GRAYSCALE)
            img = normalize_numpy(img, (-80.0, 0.0), current_range=(0.0, 255.0))
            spec = librosa.db_to_power(img)
            signal = librosa.feature.inverse.mel_to_audio(spec, audio_sampling_rate, hop_length=512, length=96000)
    
            file_file = os.path.join(os.path.dirname(file), '{}_file.wav'.format(trial_name))
            soundfile.write(file_file, signal, audio_sampling_rate)
            """
