import os
import json
from argparse import ArgumentParser

import albumentations as A
from tensorflow.keras import Sequential
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from src import ROOT_DIR
from .models import create_unet
from .functional import BCEDiceLoss, dice_coeff, IoU, IoU_Loss
from ..data.datasets import AirbusDataset


def create_parser() -> ArgumentParser:
    """Initializes parser"""
    parser = ArgumentParser()
    parser.add_argument('train_path', help='path to a .csv file with trainig data')
    parser.add_argument('val_path', help='path to a .csv file with validation data')
    parser.add_argument(
        'params_path', help='path to a .json file with hyperparameter values'
    )
    parser.add_argument('weights_path', help='where to store trained model weights')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()  # read cmd arguments

    # read hyperparameters from a .json file
    with open(os.path.join(ROOT_DIR, args.params_path), 'r') as f:
        params = json.load(f)

    # intialize hyperparameters
    BATCH_SIZE = params['batch_size']
    NUM_EPOCHS = params['num_epochs']
    LR = params['learning_rate']
    FACTOR = params['scheduler_factor']
    PAT = params['scheduler_pat']

    # initialize model
    INP_IMG_SIZE = (256, 256, 3)
    model = create_unet(INP_IMG_SIZE)

    # initialize optimizer
    optimizer = Adam(LR)

    # compile model
    model.compile(optimizer, IoU_Loss, metrics=[IoU])

    # initialize scheduler
    scheduler = ReduceLROnPlateau(factor=FACTOR, patience=PAT, verbose=1, mode='min')

    # initalize datasets
    IMG_DIR = os.path.join(ROOT_DIR, 'data/raw/train_v2/')
    PATH_TO_TRAIN = os.path.join(ROOT_DIR, args.train_path)
    transforms = A.Compose([A.Flip(), A.RandomRotate90()])
    train_dataset = AirbusDataset(
        PATH_TO_TRAIN, IMG_DIR, BATCH_SIZE, INP_IMG_SIZE[0], transforms=transforms
    )

    PATH_TO_VAL = os.path.join(ROOT_DIR, args.val_path)
    # we dont do augmentations for validation
    val_dataset = AirbusDataset(PATH_TO_VAL, IMG_DIR, BATCH_SIZE, INP_IMG_SIZE[0])
    
    # start training process
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, callbacks=[scheduler],
                        validation_data=val_dataset)

    # save model which works on full resolution
    fullres_model = Sequential()
    fullres_model.add(AveragePooling2D((3, 3)))
    fullres_model.add(model)
    fullres_model.add(UpSampling2D((3, 3)))
    fullres_model.save(os.path.join(ROOT_DIR, args.weights_path))