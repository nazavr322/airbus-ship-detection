import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove tensorflow warning messages
from argparse import ArgumentParser

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        fig.suptitle('Prediction results', fontsize=18)

        # plot orig image
        ax[0].imshow(img)
        ax[0].set_title('Original image', fontsize=14)
        ax[0].axis('off')
        ax[0].grid(False)
        
        # plot predicted mask
        mask = masks[i].numpy()
        ax[1].imshow(mask, 'binary_r')
        ax[1].set_title('Predicted mask', fontsize=14)
        ax[1].axis('off')
        ax[1].grid(False)

        # plot both image and mask
        ax[2].imshow(img)
        ax[2].imshow(mask, 'binary_r', alpha=0.4)
        ax[2].set_title('Image and mask', fontsize=14)
        ax[2].axis('off')
        ax[2].grid(False)
        fig.tight_layout()

        # save results
        filename = os.path.split(args.filenames[i])[-1].split('.')[0]
        full_fig_path = os.path.join(out_dir, filename + '-res.png')
        plt.savefig(full_fig_path, bbox_inches='tight')
        print('Results successfully saved at:', full_fig_path)
