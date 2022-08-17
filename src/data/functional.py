import numpy as np


def rle_decode(mask_rle: str, shape=(768, 768)):
    """
    Creates a binary mask from run-length-encoded string
    """
    try:
        s = mask_rle.split()
    except AttributeError:
        return np.zeros(shape, dtype=np.uint8)
    # create 2 np arrays of start indices and lenths
    starts, lengths = (np.array(x, dtype=int) for x in (s[0::2], s[1::2]))
    starts -= 1  # subtract 1 because indexing starts from 0
    ends = starts + lengths  # compute idx of last pixel using lengths
    # create flattened array of 0's to to match the indices
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1  # mask range
    return img.reshape(shape).T  # Needed to align to RLE direction