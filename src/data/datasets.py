import tensorflow as tf
from tensorboard.data.experimental import make_csv_dataset

from .functional import rle_decode


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


def get_dataset(data_path: str, batch_size: int):
    """Creates dataset from a .csv file located at `data_path`"""
    dataset = make_csv_dataset(
        data_path,
        batch_size,
        select_columns=['ImageId', 'EncodedPixels'],
        label_name='EncodedPixels'
    )
    return dataset.map(parse_batch)
