import json
import os
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageDraw



if __name__ == '__main__':
    experiment_name = 'train_2023-03-07_22-07-59 MELSIREN Beatrice Digit 0-1'
    trial_name = 'train_23bd5_00029_29_MODULATION_Type=Mult_Networks_One_Dimension_For_Each_Layer,MODULATION_hidden_features=128,MODULATION_hidden_l_2023-03-08_02-46-57'
    trial_path = os.path.join(os.getcwd(), 'Checkpoints',
                              experiment_name, trial_name)
    
    output_folder = os.path.join(os.getcwd(), "docs")
    output_filename = "{}.gif".format(trial_name)
    output_file = os.path.join(output_folder, output_filename)
    seconds_per_image = .25
    
    images = []

    def process_trial_image(index, image):
        image = image.reshape((128, 376))
        pil_image = Image.fromarray((image * 255).astype(np.uint8), "L")
        image_draw = ImageDraw.Draw(pil_image)
        image_draw.text((10, 10), 'epoch={}'.format(index+1), fill=255)
                
        image = np.array(pil_image)
        return image

    with open(os.path.join(trial_path, 'result.json'), "r") as results_file:
        for index, line in enumerate(results_file):
            json_line = json.loads(line)
            image = np.array(json_line["eval_vid"])
            image = process_trial_image(index, image)
            images.append(image)

    imageio.mimsave(output_file, images, format="GIF", duration=seconds_per_image)
