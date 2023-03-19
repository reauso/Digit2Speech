import json
import os
from typing import Optional

import cv2
import numpy as np
import imageio
from PIL import Image, ImageDraw

from util.checkpoint_helper import trial_short_name, experiment_datetime
from util.array_helper import map_numpy_values


def process_trial_image(index, image):
    image = np.squeeze(image)
    image = image.transpose((1, 2, 0)) if len(image.shape) > 2 else image
    image = map_numpy_values(image, (0, 255), (0, 1)).astype(np.uint8).copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

    text = 'epoch={}'.format(index + 1)
    font_position = (10, 3) if len(image.shape) == 3 else (10, 10)
    font_color = (0, 0, 0) if len(image.shape) == 3 else 255

    mode: Optional[str] = 'L' if len(image.shape) == 2 else 'RGB'
    pil_image = Image.fromarray((image * 255).astype(np.uint8), mode)
    image_draw = ImageDraw.Draw(pil_image)
    image_draw.text(font_position, text, fill=font_color)
    image = np.array(pil_image)

    return image


def save_gif(trial_path, save_dir, duration_per_frame):
    print('Save Gif')
    short_name = trial_short_name(os.path.basename(trial_path))
    experiment_dir = os.path.basename(os.path.dirname(trial_path))
    experiment_date = str(experiment_datetime(experiment_dir)).replace(':', '-')
    save_filename = "{}_{}.gif".format(experiment_date, short_name)
    save_path = os.path.join(save_dir, save_filename)

    images = []
    with open(os.path.join(trial_path, 'result.json'), "r") as results_file:
        for index, line in enumerate(results_file):
            json_line = json.loads(line)
            image = np.array(json_line["eval_vid"])
            image = process_trial_image(index, image)
            images.append(image)

    imageio.mimsave(save_path, images, format="GIF", duration=duration_per_frame)
