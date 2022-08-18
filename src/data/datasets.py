import os
import math

import pandas as pd
import numpy as np
import cv2 as cv
from tensorflow.keras.utils import Sequence

from .functional import rle_decode


class AirbusDataset(Sequence):
    """
    Custom dataset to iterate over .csv file.
    Returns batch of images and corresponding masks, every time __getitem__ is
    called.
    """

    def __init__(
        self,
        csv_path: str,
        target_dir: str,
        batch_size: int = 64,
        img_size: int = 768,
        transforms=None,
    ):
        self.df = pd.read_csv(csv_path)
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.img_size = (img_size, img_size)
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns amount of batches"""
        return math.ceil(len(self.df) / self.batch_size)

    def _read_img(self, filename: str) -> np.ndarray:
        """
        Reads image from file and returns it resized version of size
        self.img_size with float values in range [0, 1]
        """
        img = cv.imread(os.path.join(self.target_dir, filename))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, self.img_size)
        return np.float32(img / 255)

    def _read_mask(self, rle_str: str) -> np.ndarray:
        """
        Creates image mask of size `self.img_size` from run-length decoded
        string
        """
        mask = rle_decode(rle_str)
        mask = cv.resize(mask, self.img_size)
        return mask.astype(np.float32)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        """Returns batch of images and corresponding masks"""
        # get batch of dataframe samples starting from idx
        sub_df = self.df.iloc[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        # create batch of images
        batch_x = np.array(
            [self._read_img(filename) for filename in sub_df.ImageId]
        )
        # create batch of masks
        batch_y = np.array(
            [self._read_mask(rle_str) for rle_str in sub_df.EncodedPixels]
        )
        # if transforms is specified ...
        if self.transforms:
            # ... apply same augmentation for each image and mask in a batch
            for i, img in enumerate(batch_x):
                res = self.transforms(image=img, mask=batch_y[i])
                batch_x[i], batch_y[i] = res.values()
        return batch_x, batch_y.reshape(*batch_y.shape, -1)
