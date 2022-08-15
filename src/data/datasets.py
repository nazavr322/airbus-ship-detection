import os

import tensorflow as tf
from tensorflow.data.experimental import make_csv_dataset
from dotenv import load_dotenv

from .functional import rle_decode


load_dotenv()
RANDOM_STATE = int(os.environ['RANDOM_STATE'])


def _parse_image(filename, targ_dir=None):
    """Returns image as a tf.Tensor. `targ_dir` is a location to search for images"""
    img_dir = targ_dir if targ_dir else 'data/raw/train_v2'
    filepath = tf.strings.join([str(ROOT_DIR), img_dir, filename], separator=os.sep)
    img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(img)
    return img


def tf_rle_decode(rle_str):
    """
    Wrapper for `rle_decode` function to be compatible with tensorflow's eager execution
    """
    mask = tf.py_function(rle_decode, [rle_str], tf.uint8)
    return mask


def parse_batch(x, y):
    """
    Processes a batch of images.
    For each filename-rle_mask pair returns corresponding image array as tf.Tensor 
    """
    images = tf.map_fn(parse_image, x['ImageId'], fn_output_signature=tf.uint8)
    masks = tf.map_fn(tf_rle_decode, y, fn_output_signature=tf.uint8)
    return images, masks


def configure_for_perfomance(dataset):
    """
    Enables caching and prefetching to optimize perfomance
    """
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataset(data_path: str, batch_size: int, augmentations=None):
    """Creates dataset from a .csv file located at `data_path`"""
    dataset = make_csv_dataset(
        data_path,
        batch_size,
        select_columns=['ImageId', 'EncodedPixels'],
        label_name='EncodedPixels',
        num_epochs=1,
        shuffle_seed=RANDOM_STATE
    )
    dataset = dataset.map(parse_batch)
    if augmentations:
        dataset = dataset.map(lambda x, y: (augmentations(x), y))
    return configure_for_perfomance(dataset)
