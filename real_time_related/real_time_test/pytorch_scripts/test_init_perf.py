import os
import sys
import pdb
import json
import pickle
import numpy as np
import argparse

import build_response

def get_parser():
    parser = argparse.ArgumentParser(
        description='Unsupervised weight change')
    parser.add_argument(
            '--which_model',
            required=True, type=str,
            action='store', help='Which model to test',
            choices=build_response.ALL_MODEL_NAMES)
    parser.add_argument(
            '--which_stimuli',
            default='face', type=str,
            action='store', help='Which stimuli to test',
            choices=['face', 'objectome'])
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    all_eval_images, all_exposure_builders \
        = build_response.get_all_eval_images(
                which_stimuli=args.which_stimuli,
                im_size=build_response.get_model_im_size(args.which_model))
    all_eval_images_builders = (
        all_eval_images, all_exposure_builders)
    build_response.get_ckpt_perf(
            args.which_model,
            which_stimuli=args.which_stimuli,
            all_eval_images_builders=all_eval_images_builders)


if __name__ == '__main__':
    main()
