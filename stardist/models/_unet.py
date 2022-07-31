"""An adaption of RegNets with a more flexible stem (e.g. without strides)



from https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/regnet.py#L909-L934
"""

import warnings
import sys

from keras import backend
from keras import layers
import tensorflow as tf


def BasicConv(n_filters, strides=1, norm='bn', name=None):
    def apply(x):
        x = layers.Conv2D(
            n_filters,
            (3, 3),
            strides=strides,
            use_bias=norm != 'bn',
            padding="same",
            kernel_initializer="he_normal",
            name=None if name is None else f'{name}_conv')(x)
        if norm=='bn':
            x = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=f'{name}_bn'
            )(x)
        elif norm=='ln':
            x = layers.LayerNormalization(name=f'{name}_ln')(x)

        x = layers.ReLU(name=None if name is None else f'{name}_relu')(x)
        return x
    return apply



def DecoderUpsamplingX2Block(filters, stage, name, norm='bn',interpolation='bilinear'):
    up_name = f'{name}_decoder_stage{stage}_upsampling'
    conv1_name = f'{name}_decoder_stage{stage}a'
    conv2_name = f'{name}_decoder_stage{stage}b'
    concat_name = f'{name}_decoder_stage{stage}_concat'

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, interpolation=interpolation, name=up_name)(input_tensor)
        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
        x = BasicConv(n_filters=filters, norm=norm, name=conv1_name)(x)
        x = BasicConv(n_filters=filters, norm=norm, name=conv2_name)(x)
        return x

    return wrapper

def UnetDecoder(
        skip_layers,
        decoder_filters=(128, 64, 32, 16),
        classes=1,
        activation='sigmoid',
        norm='bn',
        name=None
):
    """ create a Unet from a backbone with given skip connection names (lowest to highest) """

    if not len(skip_layers) == len(decoder_filters):
        raise ValueError('encoder_layers and decoder_filters should have same length!') 

    prefix = "unet" if name is None else name

    def apply(backbone):

        # the lowest resolution 
        x = backbone.output

        # extract skip connections
        skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
                else backbone.get_layer(index=i).output for i in skip_layers])
        # building decoder blocks
        for i, skip in enumerate(skips):
            x = DecoderUpsamplingX2Block(decoder_filters[i], stage=i, norm=norm, name=prefix)(x, skip)

        # model head (define number of output classes)
        x = layers.Conv2D(
            filters=classes,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name=f'{name}_final_conv',
        )(x)
        x = layers.Activation(activation, name=name)(x)
        return x 

    return apply

def UnetDecoderRegnetx(classes:int =1, activation='sigmoid', norm='bn', name=None):
    return UnetDecoder(
            skip_layers=('stage_2', 'stage_1', 'stage_0', 'stem'),
            decoder_filters=(32,48,64,96),
            classes=classes,
            activation=activation,
            norm=norm,
            name=name)


if __name__ == "__main__":
    import numpy as np 

    from _regnet import RegNetX001 

    # backbone = RegNetX001(input_shape=(512,512,1), stem_strides=1, norm='ln', include_top=False)
    # backbone.summary() 
    # unet = UnetDecoderRegnetx(classes=32, activation='linear')(backbone)
    # model = tf.keras.Model(backbone.inputs, [unet])
    # x = np.zeros((1,512,512,1))
    # x[:,256,256] = 1
    # x[:,::16] = 1
    # u = model.predict(x)

    backbone = tf.keras.applications.EfficientNetV2B0(input_shape=(128,128,1), include_top=False, weights=None)