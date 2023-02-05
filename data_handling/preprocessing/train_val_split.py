from data_handling.util import files_in_directory, get_metadata_from_file_name
import pandas as pd
import os
import shutil
from tqdm import tqdm


def train_val_split(source_path, training_folder, validation_folder, split_factor=0.1):
    files = files_in_directory(
        source_path, ['**/*.wav', "**/*.flac"], recursive=True)

    metadatas = [{**get_metadata_from_file_name(
        file, as_dict=True), "file": file} for file in files]

    metadatas_df = pd.DataFrame(metadatas)
    grouped_by_speaker_lang_digit = metadatas_df.groupby(
        ["speaker", "language", "digit"])

    file_count_to_split = grouped_by_speaker_lang_digit["trial"].count(
    )*split_factor
    
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(training_folder, exist_ok=True)

    file_counts = file_count_to_split.items()

    for group, split_count in tqdm(file_counts, total=file_count_to_split.size, unit="Speaker group"):
        speaker, language, digit = group

        group_files = metadatas_df.query(f'speaker=="{speaker}" & language=="{language}" & digit=="{digit}"')[
            "file"]

        group_validation_files = group_files.sample(n=round(split_count))

        group_training_files = group_files[~group_files.index.isin(
            group_validation_files.index)]

        for file in group_validation_files:
            shutil.copy(file, os.path.join(validation_folder, os.path.basename(file)))
        for file in group_training_files:
            shutil.copy(file, os.path.join(training_folder, os.path.basename(file)))
