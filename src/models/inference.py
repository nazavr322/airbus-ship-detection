import os
from argparse import ArgumentParser

import numpy as np
import cv2 as cv


from src import ROOT_DIR
from .models import create_fullres_unet


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument(
        'weights_path', help='path to .h5 file with model weights'
    )
    parser.add_argument(
        'out_dir', help='directory where predicted masks will be stored'
    )
    parser.add_argument('filenames', nargs='+', help='path to an input image')
    return parser


def read_img(filename: str) -> np.ndarray:
    """Reads image from file, and clips its' values to be in [0, 1] range"""
    image = cv.imread(filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return np.float32(image / 255)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # read CMD arguments

    # create batch of images
    images = np.stack([
        read_img(os.path.join(ROOT_DIR, file)) for file in args.filenames
    ])

    # load model weights
    model = create_fullres_unet(os.path.join(ROOT_DIR, args.weights_path))
    # generate predictions
    masks = model(images, training=False)

    # create target folders if they don't exist
    out_dir = os.path.join(ROOT_DIR, args.out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # save predictions
    for i, img in enumerate(images):
        filename = os.path.split(args.filenames[i])[-1].split('.')[0]
        mask = masks[i].numpy()
        mask_fname = os.path.join(out_dir, filename + '-mask.png')
        cv.imwrite(mask_fname, mask)
        print('Mask successfully saved at:', mask_fname)
        img_and_mask = cv.addWeighted(img, 0.8, mask, 0.2, 0.0)
        full_img_fname = os.path.join(out_dir, filename + '-img_and_mask.png')
        cv.imwrite(full_img_fname, img_and_mask)
        print('Full image successfully saved at:', full_img_fname)
        

    

