[⬅️ Overview](../README.md)
[⬅️ Metadata](./metadata.md)

## Data

The data used for this project consists of 10,420 audio files from the Heidelberg dataset [[1]](./references.md#heidelberg-dataset) and about 6,000 newly recorded audio files.

The files from the Heidelberg dataset are in lossless FLAC format, recorded at 48 kHz and 16 Bits per sample. The dataset contains 12 speakers each speaking digits 0-9 in English and German. The length of the files vary between 0 and 2 seconds.

The newly recorded files were recorded in raw WAV at 48 kHz and 24 Bits per sample. 6 Speakers each speaking digits 0-9 in English and German in 2 second snippets. They can be downloaded from [Audioaufnahmen](https://fhd-my.sharepoint.com/:f:/g/personal/rene_ebertowski_hs-duesseldorf_de/Eg08lVFc4aFFh79xyDdjhDgByM43i7pD2KxiGpu1O4Ol-w?e=Qbpiep).

For easier access as well as training and validation purposes, the data is abstracted into two pytorch `torch.utils.data.Dataset`s. The dataset classes are split into `DigitAudioDatasetForSignal` and `DigitAudioDatasetForSpectrograms`, both defined in [Dataset.py](../data_handling/Dataset.py) and each providing audio signal and spectrogram images respectively. Furthermore, they both expose tensorized and raw metadata.

[➡️ Preprocessing](./preprocessing.md)
