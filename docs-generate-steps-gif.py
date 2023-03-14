import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import imageio
from PIL import Image, ImageDraw

from data_handling.util import latest_experiment_path, best_trial_path, map_numpy_values


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


if __name__ == '__main__':
    # defaults for config
    checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    save_path = os.path.join(os.getcwd(), 'Docs')

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path(checkpoint_dir), help='Dir with all experiments.')
    parser.add_argument("--save_dir", type=Path, default=Path(save_path), help='Saving dir for generated Audio.')
    parser.add_argument("--experiment", type=str, default='latest', help='Name of the experiment of the model or '
                                                                         'latest for automatic detection.')
    parser.add_argument("--trial", type=str, default='best', help='Trial name or best for automatic detection')
    parser.add_argument("--seconds_per_image", type=float, default=0.25, help='Seconds between frames.')
    args = parser.parse_args()

    # automatic detections
    args.experiment = os.path.basename(
        latest_experiment_path(args.checkpoint_dir)) if args.experiment == 'latest' else args.experiment
    experiment_dir = os.path.join(args.checkpoint_dir, args.experiment)
    args.trial = os.path.basename(best_trial_path(experiment_dir)) if args.trial == 'best' else args.trial

    # define necessary paths
    trial_path = os.path.join(args.checkpoint_dir, args.experiment, args.trial)
    print('Use Trial at location: {}'.format(trial_path))

    save_filename = "{}.gif".format(args.trial)
    save_file = os.path.join(args.save_dir, save_filename)
    print('Save at location: {}'.format(save_file))

    images = []
    with open(os.path.join(trial_path, 'result.json'), "r") as results_file:
        for index, line in enumerate(results_file):
            json_line = json.loads(line)
            image = np.array(json_line["eval_vid"])
            image = process_trial_image(index, image)
            images.append(image)

    imageio.mimsave(save_file, images, format="GIF", duration=args.seconds_per_image)
