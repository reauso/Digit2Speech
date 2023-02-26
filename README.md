# Digit2Speech

This project is an attempt to create a deep neural network which generated spoken digits. Primarily, it acts as a way to experiment with different neural network architectures.

## Metadata

The audio should be generated from the following metadata:

- digit
- language
- gender
- tone height (mfcc)

Given some metadata, the neural network should be able to generate a fitting audio file.

## Concepts and architectures

We evaluated different concepts and architectues to analyze and generate audio data. The following list contains the concepts and architectures we tried.

- Siren (trained on raw signal, ouputs raw signal)
- Siren (trained on mel spectrogram, outputs mel spectrogram)
- GAN (trained on mel spectogram, outputs mel spectrogram)

### Siren (raw signal)

The first concept was to use the raw signal as a base for the audio data. The Siren architecture enables to train a neural network on raw signal data (i.e. sample amplitudes at given times) based on the metadata and additionally the position. We were hoping for the neural network to associate the metadata with the position and value of the samples.

### Siren (mel spectrogram)

### GAN (mel spectrogram)

## Data

The data for the training and validation consists of ~10.000 audio files from the heisenberg dataset and ~6.000 newly recorded audio files done by us. In order to have all necessary and consistent data, some preprocessing steps are obligatory.

- Split the audio files into 2 second chunks, having the spoken digit centered in the chunk,
- generate mfcc coefficients from the audio files,
- generate mel spectograms from the audio files,
- split the data into training and validation data,
- extract the metadata from the file name.
