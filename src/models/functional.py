import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy


def tversky_coef(
    y_true: tf.Tensor, y_pred: tf.Tensor, beta: float = .5, smooth: float = 1.
) -> tf.Tensor:
    """
    Tversky coefficient is a generalization of the Dice's coefficient. 
    When β=1/2, Tversky coefficient is equal to the Dice's coefficient.
    """
    axis_to_reduce = range(1, K.ndim(y_pred))  # don't include batch axis
    numerator = K.sum(y_true * y_pred, axis=axis_to_reduce)  # intersection
    # union
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    denominator = K.sum(denominator, axis=axis_to_reduce)
    return (numerator + smooth) / (denominator + smooth) 


def BCEDiceLoss(
    y_true: tf.Tensor, y_pred: tf.Tensor, beta: float = .5, smooth: float = 1.
) -> tf.Tensor:
    """Combination of binary cross entropy and dice losses"""
    bce = binary_crossentropy(y_true, y_pred)
    axis_to_reduce = range(1, K.ndim(bce))  
    bce = K.mean(bce, axis=axis_to_reduce)

    dice_coefficient = tversky_coef(y_true, y_pred, 0.5, smooth)

    return beta * (1. - dice_coefficient) + (1. - beta) * bce


def iou_score(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-10) -> tf.Tensor:
    """Computes IoU score"""
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)


def IoU_Loss(y_true, y_pred):
    """Computes loss using IoU score"""
    return 1 - iou_score(y_true, y_pred)