## Data

The data for the training and validation consists of ~10.000 audio files from the heisenberg dataset and ~6.000 newly recorded audio files done by us. In order to have all necessary and consistent data, some preprocessing steps are obligatory. This package contains a script to do the following steps:

- Split the audio files into 2 second chunks, having the spoken digit centered in the chunk,
- generate mfcc coefficients from the audio files,
- generate mel spectograms from the audio files,
- split the data into training and validation data,
- extract the metadata from the file name.

For easier access and training and validation purposes, the data is abstracted into a pytorch `torch.utils.data.Dataset`.
