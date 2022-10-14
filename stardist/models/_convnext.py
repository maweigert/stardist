# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""ConvNeXt models for Keras.

References:

- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  (CVPR 2022)
"""

import numpy as np
import tensorflow.compat.v2 as tf
import sys
import warnings
from keras import backend
from keras import layers
from keras import models
from keras import utils
from keras.applications import imagenet_utils
from keras.engine import sequential
from keras.engine import training as training_lib


MODEL_CONFIGS = {
    "mini": {
        # "depths": [3,3,3,3,3],
        # "projection_dims": [64, 96, 128, 196, 256],
        "depths": [3,3,3,3],
        "projection_dims": [64, 96, 128, 196],
        "default_size": 224,
    }
}

class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It performs batch-wise dropping rather than sample-wise. In libraries like
    `timm`, it's similar to `DropPath` layers that drops residual paths
    sample-wise.

    References:
      - https://github.com/rwightman/pytorch-image-models

    Args:
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].

    Returns:
      Tensor either with the residual path dropped or kept.
    """

    def __init__(self, drop_path_rate, **kwargs):
        super().__init__(**kwargs)
        self.drop_path_rate = drop_path_rate

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_path_rate": self.drop_path_rate})
        return config


class LayerScale(layers.Layer):
    """Layer scale module.

    References:
      - https://arxiv.org/abs/2103.17239

    Args:
      init_values (float): Initial value for layer scale. Should be within
        [0, 1].
      projection_dim (int): Projection dimensionality.

    Returns:
      Tensor multiplied to the scale.
    """

    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,))
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def ConvNeXtBlock(
    projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, name=None
):
    """ConvNeXt block.

    References:
    - https://arxiv.org/abs/2201.03545
    - https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Notes:
      In the original ConvNeXt implementation (linked above), the authors use
      `Dense` layers for pointwise convolutions for increased efficiency.
      Following that, this implementation also uses the same.

    Args:
      projection_dim (int): Number of filters for convolution layers. In the
        ConvNeXt paper, this is referred to as projection dimension.
      drop_path_rate (float): Probability of dropping paths. Should be within
        [0, 1].
      layer_scale_init_value (float): Layer scale value. Should be a small float
        number.
      name: name to path to the keras layer.

    Returns:
      A function representing a ConvNeXtBlock block.
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(inputs):
        x = inputs


        if sys.platform=='darwin':
            warnings.warn('DARWIN!')
            group_dim = 1
        else: 
            group_dim = projection_dim

        x = layers.Conv2D(
            filters=projection_dim,
            kernel_size=7,
            padding="same",
            groups=group_dim,
            name=name + "_depthwise_conv",
        )(x)

        x = layers.LayerNormalization(epsilon=1e-6, name=name + "_layernorm")(x)
        x = layers.Dense(2 * projection_dim, name=name + "_pointwise_conv_1")(x)
        x = layers.Activation("relu", name=name + "_gelu")(x)
        x = layers.Dense(projection_dim, name=name + "_pointwise_conv_2")(x)

        if layer_scale_init_value is not None:
            x = LayerScale(
                layer_scale_init_value,
                projection_dim,
                name=name + "_layer_scale",
            )(x)
        if drop_path_rate:
            layer = StochasticDepth(
                drop_path_rate, name=name + "_stochastic_depth"
            )
        else:
            layer = layers.Activation("linear", name=name + "_identity")

        return inputs + layer(x)

    return apply


def PreStem(name=None):
    """Normalizes inputs with ImageNet-1k mean and std.

    Args:
      name (str): Name prefix.

    Returns:
      A presemt function.
    """
    if name is None:
        name = "prestem" + str(backend.get_uid("prestem"))

    def apply(x):
        x = layers.Normalization(
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            variance=[
                (0.229 * 255) ** 2,
                (0.224 * 255) ** 2,
                (0.225 * 255) ** 2,
            ],
            name=name + "_prestem_normalization",
        )(x)
        return x

    return apply


def Head(num_classes=1000, name=None):
    """Implementation of classification head of RegNet.

    Args:
      num_classes: number of classes for Dense layer
      name: name prefix

    Returns:
      Classification head function.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.LayerNormalization(
            epsilon=1e-6, name=name + "_head_layernorm"
        )(x)
        x = layers.Dense(num_classes, name=name + "_head_dense")(x)
        return x

    return apply


def ConvNeXt(
    depths,
    projection_dims,
    drop_path_rate=0.0,
    layer_scale_init_value=1e-6,
    default_size=224,
    model_name="convnext",
    include_preprocessing=True,
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates ConvNeXt architecture given specific configuration.

    Args:
      depths: An iterable containing depths for each individual stages.
      projection_dims: An iterable containing output number of channels of
      each individual stages.
      drop_path_rate: Stochastic depth probability. If 0.0, then stochastic
        depth won't be used.
      layer_scale_init_value: Layer scale coefficient. If 0.0, layer scaling
        won't be used.
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_preprocessing: boolean denoting whther to include preprocessing in
        the model. When `weights="imagenet"` this should be always set to True.
        But for other models (e.g., randomly initialized) users should set it
        to False and apply preprocessing to data accordingly.
      include_top: Boolean denoting whether to include classification head to
        the model.
      weights: one of `None` (random initialization), `"imagenet"` (pre-training
        on ImageNet-1k), or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
        use as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top`
        is False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top`
        is `False`.
        - `None` means that the output of the model will be the 4D tensor output
          of the last convolutional layer.
        - `avg` means that global average pooling will be applied to the output
          of the last convolutional layer, and thus the output of the model will
          be a 2D tensor.
        - `max` means that global max pooling will be applied.
      classes: optional number of classes to classify images into, only to be
        specified if `include_top` is True, and if no `weights` argument is
        specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax`, or `None`
          when using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000
          when using ImageNet.
    """

    # Determine proper input shape.
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if input_tensor is not None:
        inputs = utils.layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = inputs
    if include_preprocessing:
        channel_axis = (
            3 if backend.image_data_format() == "channels_last" else 1
        )
        num_channels = input_shape[channel_axis - 1]
        if num_channels == 3:
            x = PreStem(name=model_name)(x)

    # Stem block.
    stem = sequential.Sequential(
        [
            layers.Conv2D(
                projection_dims[0],
                kernel_size=5,
                strides=1,
                padding='same',
                name=model_name + "_stem_conv",
            ),
            layers.LayerNormalization(
                epsilon=1e-6, name=model_name + "_stem_layernorm"
            ),
        ],
        name=model_name + "_stem",
    )

    # Downsampling blocks.
    downsample_layers = []
    downsample_layers.append(stem)

    num_downsample_layers = len(depths)-1
    for i in range(num_downsample_layers):
        downsample_layer = sequential.Sequential(
            [
                layers.LayerNormalization(
                    epsilon=1e-6,
                    name=model_name + "_downsampling_layernorm_" + str(i),
                ),
                layers.Conv2D(
                    projection_dims[i + 1],
                    kernel_size=2,
                    strides=2,
                    name=model_name + "_downsampling_conv_" + str(i),
                ),
            ],
            name=model_name + "_downsampling_block_" + str(i),
        )
        downsample_layers.append(downsample_layer)

    # Stochastic depth schedule.
    # This is referred from the original ConvNeXt codebase:
    # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
    depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(depths))
    ]

    # First apply downsampling blocks and then apply ConvNeXt stages.
    cur = 0

    num_convnext_blocks = len(depths)
    for i in range(num_convnext_blocks):
        x = downsample_layers[i](x)
        for j in range(depths[i]):
            x = ConvNeXtBlock(
                projection_dim=projection_dims[i],
                drop_path_rate=depth_drop_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value,
                name=model_name + f"_stage_{i}_block_{j}",
            )(x)
        cur += depths[i]

    if include_top:
        x = Head(num_classes=classes, name=model_name)(x)
        imagenet_utils.validate_activation(classifier_activation, weights)

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    model = training_lib.Model(inputs=inputs, outputs=x, name=model_name)


    return model



def preprocess_input(x, data_format=None):
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the convnext model
    implementation. Users are no longer required to call this method to
    normalize the input data. This method does nothing and only kept as a
    placeholder to align the API surface between old and new version of model.

    Args:
      x: A floating point `numpy.array` or a `tf.Tensor`.
      data_format: Optional data format of the image tensor/array. Defaults to
        None, in which case the global setting
        `tf.keras.backend.image_data_format()` is used (unless you changed it,
        it defaults to "channels_last").{mode}

    Returns:
      Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


## Instantiating variants ##


def ConvNeXtTiny(
    model_name="convnext_tiny",
    include_top=True,
    include_preprocessing=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return ConvNeXt(
        depths=MODEL_CONFIGS["tiny"]["depths"],
        projection_dims=MODEL_CONFIGS["tiny"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["tiny"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=include_preprocessing,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def ConvNeXtMini(
    model_name="convnext_mini",
    include_top=True,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    return ConvNeXt(
        depths=MODEL_CONFIGS["mini"]["depths"],
        projection_dims=MODEL_CONFIGS["mini"]["projection_dims"],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        default_size=MODEL_CONFIGS["mini"]["default_size"],
        model_name=model_name,
        include_top=include_top,
        include_preprocessing=False,
        weights=None,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )




##### UNET 
##### https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/models/unet.py



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
            print(x.shape, skip.shape)
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

def UnetDecoderConvNext(classes:int =1, activation='sigmoid', norm='bn', name=None):
    return UnetDecoder(
            # skip_layers=('convnext_mini_stage_3_block_2_identity', 'convnext_mini_stage_2_block_2_identity', 'convnext_mini_stage_1_block_2_identity', 'convnext_mini_stage_0_block_0_depthwise_conv'),
            skip_layers=('convnext_mini_stage_2_block_2_identity', 'convnext_mini_stage_1_block_2_identity', 'convnext_mini_stage_0_block_0_depthwise_conv'),
            # skip_layers=('convnext_mini_stage_2_block_2_identity', 'convnext_mini_stage_1_block_2_identity', 'convnext_mini_stage_0_block_0_identity'),
            decoder_filters=(48,64,96),
            classes=classes,
            activation=activation,
            norm=norm,
            name=name)

def UnetConvNext(input_shape =(None,None,1), classes:int =1, activation='sigmoid', norm='bn', name=None):
    model = ConvNeXtMini(include_top=False, input_shape=input_shape)
    unet = models.Model(model.input, UnetDecoderConvNext(classes=classes, activation=activation,norm=norm, name=name)(model))
    return unet 


if __name__ == "__main__":
    model = ConvNeXtMini(include_top=False, input_shape=(512,512,1))

    model = UnetConvNext(input_shape=(None,None,1), activation='linear')