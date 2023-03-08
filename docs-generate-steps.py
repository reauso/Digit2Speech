import json
import os
import cv2
from moviepy import editor
import numpy as np
from data_handling.util import print_numpy_stats
import imageio
from tqdm import tqdm

if __name__ == '__main__':

    experiment_name = 'train_2023-03-07_22-07-59 MELSIREN Beatrice Digit 0-1'
    trial_name = 'train_23bd5_00029_29_MODULATION_Type=Mult_Networks_One_Dimension_For_Each_Layer,MODULATION_hidden_features=128,MODULATION_hidden_l_2023-03-08_02-46-57'
    trial_path = os.path.join(os.getcwd(), 'Checkpoints',
                              experiment_name, trial_name)

    images = []

    with open(os.path.join(trial_path, 'result.json'), "r") as results_file:
        for index, line in enumerate(results_file):
            json_line = json.loads(line)
            image = np.array(json_line["eval_vid"])
            image = image.reshape((128, 376)).astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # image = cv2.addText(image, 'epoch={}'.format(index+1), (10,10), nameFont="Calibri")

            images.append(image.astype(np.uint8))

    with imageio.get_writer("steps.gif", mode="I") as writer:
        for frame in tqdm(images, desc="Generating GIF with images", unit="Images"):
            writer.append_data(frame)
