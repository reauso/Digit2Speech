import argparse
import os
from pathlib import Path

from util.checkpoint_helper import latest_experiment_path, best_trial_path
from docs.trial_gif import save_gif

if __name__ == '__main__':
    # defaults for config
    checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoints')
    save_path = os.path.join(os.getcwd(), 'Docu')

    # config
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=Path(checkpoint_dir), help='Dir with all experiments.')
    parser.add_argument("--save_dir", type=Path, default=Path(save_path), help='Saving dir for generated Audio.')
    parser.add_argument("--experiment", type=str, default='latest', help='Name of the experiment of the model or '
                                                                         'latest for automatic detection.')
    parser.add_argument("--trial", type=str, default='best', help='Trial name or best for automatic detection.')
    parser.add_argument('--all', action='store_true', help='Applies all steps available in this pipeline.')
    parser.add_argument('--gif', action='store_true', help='Generates a gif from the evaluation vid images from '
                                                           'tensorboard of the given trial.')
    parser.add_argument("--gif_dpf", type=float, default=0.25, help='Duration per frame in seconds.')
    args = parser.parse_args()

    # automatic detections
    args.experiment = os.path.basename(
        latest_experiment_path(args.checkpoint_dir)) if args.experiment == 'latest' else args.experiment
    experiment_dir = os.path.join(args.checkpoint_dir, args.experiment)
    args.trial = os.path.basename(best_trial_path(experiment_dir)) if args.trial == 'best' else args.trial

    # define necessary paths
    trial_path = os.path.join(args.checkpoint_dir, args.experiment, args.trial)
    print('Use Trial at location: {}'.format(trial_path))

    # apply pipeline steps
    if args.all or args.gif:
        save_gif(trial_path, args.save_dir, args.gif_dpf)
