[⬅️ Overview](../README.md)
[⬅️ Preprocessing](./preprocessing.md)

# Concepts and architectures

We evaluated different concepts and architectures to analyze and generate audio data. The following list contains the concepts and architectures we tried.

- Siren (trained on raw signal, ouputs raw signal)
- Siren (trained on mel spectrogram, outputs mel spectrogram)
- GAN (trained on mel spectogram, outputs mel spectrogram)

The architectures are further explained in the [model](./model/README.md) subpage.

In this part of the documentation, the considered network architectures are presented and further discussed.

## Siren (raw signal)<a name="siren_signal"></a>

The first concept was to use the raw signal as a base for the audio data. The Siren architecture enables to train a neural network on raw signal data (i.e. sample amplitudes at given times) and finally predict a signal for given a time in a signal. Additionally, a mapping layer (FiLm) based on the metadata and additionally the position. We were hoping for the neural network to associate the metadata with the position and value of the samples.

<figure>
  <img
  id="figures-siren-audio"
  src="./figures-siren-audio.png"
  alt="">
  <figcaption>Fig. 3: High level architecture of Siren signal generator(based on TODO)</figcaption>
</figure>

## Siren (Mel spectrogram)<a name="siren_mel"></a>

The second concept was to use the Siren architecture but with images displaying Mel spectrograms of the audio files as the base for generation.

<figure>
  <img
  id="figures-siren-mel"
  src="./figures-siren-mel.png"
  alt="">
  <figcaption>Fig. 4: High level architecture of Siren Mel generator (based on TODO)</figcaption>
</figure>
<figure>
  <img
  id="figures-siren-mel-process"
  src="./train_23bd5_00029_29_MODULATION_Type=Mult_Networks_One_Dimension_For_Each_Layer,MODULATION_hidden_features=128,MODULATION_hidden_l_2023-03-08_02-46-57.gif"
  alt="Development of a model training from epoch 0 to 50. It is visible how the quality of the generated image increases from a blurry spectrogram to a more clear one.">
  <figcaption>Fig. 5: Development of an training attempt of a single language, single gender, 2 digit generator in 50 epochs</figcaption>
</figure>

## GAN (mel spectrogram)<a name="gan_mel"></a>

[➡️ Development and usage](./development-and-usage.md)
