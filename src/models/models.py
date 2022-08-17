import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import (
    UpSampling2D, concatenate, Conv2D, BatchNormalization, MaxPool2D, ReLU, AveragePooling2D
)
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input 


def _get_skip_connections(model):
    """Returns a list of a layers, which outputs need to be forwarded to decoder"""
    skip_connections = []
    # add input layer and first activation
    skip_connections.append(model.get_layer('tf.nn.bias_add'))
    skip_connections.append(model.get_layer('conv1_relu'))

    for i, layer in enumerate(model.layers[:-1]):
        if layer.name.endswith('_out'):
            next_layer = model.layers[i + 1]
            if next_layer.input_shape[1] != next_layer.output_shape[1]:
                skip_connections.append(layer)
    return skip_connections


def _add_upsamling_block(model, previous_layer, encoder_layer):
    """
    Returns a model with upsampling block added, whose inputs is an activations from a
    previous layer and activations from a corresponding layer in encoder.
    """
    prev_activations = previous_layer.output
    try:
        encoder_activations = encoder_layer.output
    except AttributeError:
        encoder_activations = encoder_layer

    # each block halves amount of filters and doubles the height and width
    num_filters = round(prev_activations.shape[-1] / 2)
    up = UpSampling2D()(prev_activations)  # double height and width
    concat = concatenate([up, encoder_activations])  # concatenate 2 embeddings
    # halve amount of filters
    conv_1 = Conv2D(num_filters, 3, padding='same', activation='relu')(concat)
    conv_2 = Conv2D(num_filters, 3, padding='same', activation='relu')(conv_1)
    return Model(inputs=model.input, outputs=conv_2)


def create_unet50(
    input_shape: tuple[int], weights: str = None
) -> Model:
    """Creates a UNet acrhitecture with pretrained ResNet50 encoder"""
    # process image to be compatible with ResNet
    inp = Input(input_shape, dtype=tf.float32)
    inp = preprocess_input(inp)
    
    # if weights are provided, don't load pretrained resnet
    resnet_weights = None if weights else 'imagenet'
    
    # create and freeze encoder
    model = ResNet50(False, resnet_weights, input_tensor=inp, input_shape=input_shape)
    model.trainable = False

    # get skip connections
    skip_connections = _get_skip_connections(model)
    
    # create decoder
    for layer in reversed(skip_connections):
        model = _add_upsamling_block(model, model.layers[-1], layer)

    # add 1x1 convolution with 1 filter and sigmoid to convert embeddings to masks
    final_conv = Conv2D(1, 1, activation='sigmoid')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=final_conv)

    # load weights from file if provided
    if weights:
        model.load_weights(weights)
    return model


def _double_conv_block(x, n_filters):
    for _ in range(2):
        x = Conv2D(n_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    return x


def create_unet(input_shape: tuple[int, ...], weights=None) -> Model:
    """Creates model with UNet architecture"""
    inp = Input(input_shape, dtype=tf.float32)
    x = BatchNormalization()(inp)

    skip_connections = []  # we'll store activations here
    
    # create encoder
    for n_filters in (32, 64, 128, 256):
        x = _double_conv_block(x, n_filters)
        skip_connections.append(x)
        x = MaxPool2D()(x)
    
    # create bottleneck
    model = Model(inputs=inp, outputs=_double_conv_block(x, 512))

    # create_decoder
    for layer in reversed(skip_connections):
        model = _add_upsamling_block(model, model.layers[-1], layer)
    
    # add 1x1 convolution with 1 filter and sigmoid to convert embeddings to masks
    final_conv = Conv2D(1, 1, activation='sigmoid')(model.layers[-1].output)
    model = Model(inputs=model.input, outputs=final_conv)

    # load weights from file if provided
    if weights:
        model.load_weights(weights)
    return model


def create_fullres_unet(path_to_weights: str):
    """
    Creates unet model that is compatible with original image size from dataset
    """
    model = Sequential()
    model.add(AveragePooling2D((3, 3), input_shape=(768, 768, 3)))
    model.add(create_unet((256, 256, 3)))
    model.add(UpSampling2D((3, 3)))
    model.load_weights(path_to_weights)
    return model