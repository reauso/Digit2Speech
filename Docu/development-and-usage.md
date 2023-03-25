[⬅️ Overview](../README.md)
[⬅️ Concepts and architectures](./concepts-and-architectures.md)

# Development and usage

The project is structured as follows:

- The `data_handling` folder is a python module to expose the dataset and data loaders, providing functions for pre-processing and some utilities.
- The `model` folder contains the neural network implementations.
- The `docs` folder contains the markdown and more files for documentation.
- The `analysis` folder contains the Scripts for an Experiment Analysis.
- In the root project folder, there are some python files with main functions:
  - `preprocessing.py` runs the required preprocessing steps
  - `train_siren_mel.py` and `train_siren_signal.py` each start a training process for a Mel Siren and signal Siren respectively.
  - Likewise, `inference_mel.py` and `inference_signal.py` both generate images and audio based on validation files using pre trained checkpoints.
  - The `ray_result_analyzer.py` Scipt starts a manual Analysis of an Experiment and shows all Trials with the resulting metrics and the influence of each config to the metric scores.

To run any code, first install all required dependencies with `pip install -r requirements.txt`.

## Data

For both training and inference (at least using validation files), data inside a folder `Dataset` is needed. The audio files for this project are located at [Audioaufnahmen](https://fhd-my.sharepoint.com/:f:/g/personal/rene_ebertowski_hs-duesseldorf_de/Eg08lVFc4aFFh79xyDdjhDgByM43i7pD2KxiGpu1O4Ol-w?e=Qbpiep).

- Create a folder `Dataset` at root level and paste all files there.
- Run preprocessing with `python preprocessing.py --split_train_val --mfcc --spectrogram --pad` and wait for it to finish. For help what each flag does and usage reference run `python preprocessing.py --help`.
- Training and validation files should now be in [`Dataset/training`](../Dataset/training) and [`Dataset/validation`](../Dataset/validation) with mfccs and spectrograms

## Training

Please note that for training a model, training and validation files are needed.

Run `python train_siren_mel.py` or `python train_siren_signal.py`. This starts ray and trains multiple configurations at once.

## Inference

To generate audio files using pre-trained models, first the models have to be downloaded and put into a folder.

- Head to [Modelle](https://fhd-my.sharepoint.com/:f:/g/personal/rene_ebertowski_hs-duesseldorf_de/EqPO7QWZeeZDtL7zB9GIKs4BTi4E4g4ND8qSA4WZaXJnuA?e=ZBoT5Y) and download the model you want to try out.
- Extract the archive.
- Create a folder `Checkpoints` and paste the folder from the archive in it.
- Run `python inference_mel.py` or `python inference_signal.py`. You can also pass in custom directories, see usage reference with `python inference_mel.py --help` or `python inference_signal.py --help` for more info.
- Per default generated files are saved to a new folder `GeneratedAudio`.

## Analysis

Run 'python ray_result_analyzer.py' to create an Analysis of one or multiple Experiment. Note that this Script does not support the argparse package so that you have to configurate it in the Script itself.

[➡️ References](./references.md)
