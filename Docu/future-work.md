[⬅️ Overview](../README.md)
[⬅️ Development and usage](./development-and-usage.md)

# Signal Siren

The signal Siren approach needs a more detailed analysis of how to reduce noise and how to control the synthesis 
process, so that we can synthesize multiple speakers with different styles and all 10 digits.

# Mel Siren

There are two improvements for Future Work that we suggest. At first, we need to gain more control over the synthesis
process. Our presented audio files in the [Concepts and architectures](./concepts-and-architectures.md) section show, 
that at the current state the model sometimes ignores the given metadata and does not synthesize the desired digit.
Secondly we need to improve the conversion from a mel-spectrogram into the audio signal. For Future work we
could utilize MelGAN [[7]](./references.md#papers-melgan) for this conversion instead of rely on lossy mathematical
calculations.

# GANs

We also want to try other techniques and model architectures to create the audio signal from our metadata like GANs.
One approach could be StyleGAN [[8]](./references.md#papers-stylegan)[[9]](./references.md#papers-stylegan2), another
to convert the metadata-to-spectrogram task to an image-to-image task which has already been better researched.
Then we could utilize models like Pix2Pix [[10]](./references.md#papers-p2p) or Pix2PixHD 
[[11]](./references.md#papers-p2phd) to generate the spectrograms.


[➡️ References](./references.md)