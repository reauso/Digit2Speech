# Concepts and architectures

We evaluated different concepts and architectures to analyze and generate audio data. The following list contains the concepts and architectures we tried.

- Siren (trained on raw signal, ouputs raw signal)
- Siren (trained on mel spectrogram, outputs mel spectrogram)
- GAN (trained on mel spectogram, outputs mel spectrogram)

The architectures are further explained in the [model](./model/README.md) subpage.

In this part of the documentation, the considered network architectures are presented and further discussed.

## Siren (raw signal)<a name="siren_signal"></a>

The first concept was to use the raw signal as a base for the audio data. The Siren architecture enables to train a neural network on raw signal data (i.e. sample amplitudes at given times) based on the metadata and additionally the position. We were hoping for the neural network to associate the metadata with the position and value of the samples.

## Siren (mel spectrogram)<a name="siren_mel"></a>

## GAN (mel spectrogram)<a name="gan_mel"></a>
