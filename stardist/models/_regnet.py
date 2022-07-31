"""An adaption of RegNets with a more flexible stem (e.g. without strides)



from https://github.com/keras-team/keras/blob/v2.9.0/keras/applications/regnet.py#L909-L934
"""

import warnings
import sys

from keras import backend
from keras import layers
from keras.engine import training
from keras.utils import layer_utils
import tensorflow as tf


MODEL_CONFIGS = {
    "x001": {
        "depths": [1, 1, 3, 5],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "X",
    },
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "X",
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X",
    },
}


def Stem(name=None, n_filters: int = 32, strides: int = 2, norm='bn'):
    if name is None:
        name = "stem" + str(backend.get_uid("stem"))
    
    def apply(x):
        x = layers.Conv2D(
            n_filters,
            (3, 3),
            strides=strides,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_stem_conv",
        )(x)
        if norm=='bn':
            x = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_stem_bn"
            )(x)
        elif norm=='ln':
            x = layers.LayerNormalization(name=name + "_stem_ln")(x)

        x = layers.ReLU(name=name + "_stem_relu")(x)
        return x

    return apply


def SqueezeAndExciteBlock(filters_in, se_filters, name=None):
    """Implements the Squeeze and excite block (https://arxiv.org/abs/1709.01507).
    Args:
      filters_in: input filters to the block
      se_filters: filters to squeeze to
      name: name prefix
    Returns:
      A function object
    """
    if name is None:
        name = str(backend.get_uid("squeeze_and_excite"))

    def apply(inputs):
        x = layers.GlobalAveragePooling2D(
            name=name + "_squeeze_and_excite_gap", keepdims=True
        )(inputs)
        x = layers.Conv2D(
            se_filters,
            (1, 1),
            activation="relu",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_squeeze",
        )(x)
        x = layers.Conv2D(
            filters_in,
            (1, 1),
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=name + "_squeeze_and_excite_excite",
        )(x)
        x = tf.math.multiply(x, inputs)
        return x

    return apply


def XBlock(filters_in, filters_out, group_width, stride=1, norm='bn', name=None):
    """Implementation of X Block.
    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("xblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must "
                f"be equal for stride={stride}."
            )

        # Declare layers
        groups = filters_out // group_width

        if stride != 1:
            skip = layers.Conv2D(
                filters_out,
                (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1",
            )(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn"
            )(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        if norm=='bn':
            x = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1__bn"
            )(x)
        elif norm=='ln':
            x = layers.LayerNormalization(name=name + "_conv_1x1_1__ln")(x)

        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        if sys.platform=='darwin':
            warnings.warn('setting groups to 1 on MAC!')
            groups = 1

        # conv_3x3
        x = layers.Conv2D(
            filters_out,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        if norm=='bn':
            x = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3__bn"
            )(x)
        elif norm=='ln':
            x = layers.LayerNormalization(name=name + "_conv_3x3__ln")(x)

        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        if norm=='bn':
            x = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
            )(x)
        elif norm=='ln':
            x = layers.LayerNormalization(name=name + "_conv_1x1_2_ln")(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


def YBlock(
    filters_in, filters_out, group_width, stride=1, squeeze_excite_ratio=0.25, name=None
):
    """Implementation of Y Block.
    Reference: [Designing Network Design
    Spaces](https://arxiv.org/abs/2003.13678)
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("yblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out}) "
                f"are not equal for stride {stride}. Input and output filters must  "
                f"be equal for stride={stride}."
            )

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        if stride != 1:
            skip = layers.Conv2D(
                filters_out,
                (1, 1),
                strides=stride,
                use_bias=False,
                kernel_initializer="he_normal",
                name=name + "_skip_1x1",
            )(inputs)
            skip = layers.BatchNormalization(
                momentum=0.9, epsilon=1e-5, name=name + "_skip_bn"
            )(skip)
        else:
            skip = inputs

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_1x1_1_relu")(x)

        # conv_3x3
        x = layers.Conv2D(
            filters_out,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn"
        )(x)
        x = layers.ReLU(name=name + "_conv_3x3_relu")(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(filters_out, se_filters, name=name)(x)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
        )(x)

        x = layers.ReLU(name=name + "_exit_relu")(x + skip)

        return x

    return apply


def ZBlock(
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    bottleneck_ratio=0.25,
    name=None,
):
    """Implementation of Z block Reference: [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877).
    Args:
      filters_in: filters in the input tensor
      filters_out: filters in the output tensor
      group_width: group width
      stride: stride
      squeeze_excite_ratio: expansion ration for Squeeze and Excite block
      bottleneck_ratio: inverted bottleneck ratio
      name: name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("zblock"))

    def apply(inputs):
        if filters_in != filters_out and stride == 1:
            raise ValueError(
                f"Input filters({filters_in}) and output filters({filters_out})"
                f"are not equal for stride {stride}. Input and output filters must be"
                f" equal for stride={stride}."
            )

        groups = filters_out // group_width
        se_filters = int(filters_in * squeeze_excite_ratio)

        inv_btlneck_filters = int(filters_out / bottleneck_ratio)

        # Build block
        # conv_1x1_1
        x = layers.Conv2D(
            inv_btlneck_filters,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_1",
        )(inputs)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_1_bn"
        )(x)
        x = tf.nn.silu(x)

        # conv_3x3
        x = layers.Conv2D(
            inv_btlneck_filters,
            (3, 3),
            use_bias=False,
            strides=stride,
            groups=groups,
            padding="same",
            kernel_initializer="he_normal",
            name=name + "_conv_3x3",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_3x3_bn"
        )(x)
        x = tf.nn.silu(x)

        # Squeeze-Excitation block
        x = SqueezeAndExciteBlock(inv_btlneck_filters, se_filters, name=name)

        # conv_1x1_2
        x = layers.Conv2D(
            filters_out,
            (1, 1),
            use_bias=False,
            kernel_initializer="he_normal",
            name=name + "_conv_1x1_2",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_conv_1x1_2_bn"
        )(x)

        if stride != 1:
            return x
        else:
            return x + inputs

    return apply


def Stage(block_type, depth, group_width, filters_in, filters_out, norm='bn', name=None):
    """Implementation of Stage in RegNet.
    Args:
      block_type: must be one of "X", "Y", "Z"
      depth: depth of stage, number of blocks to use
      group_width: group width of all blocks in  this stage
      filters_in: input filters to this stage
      filters_out: output filters from this stage
      name: name prefix
    Returns:
      Output tensor of Stage
    """
    if name is None:
        name = str(backend.get_uid("stage"))

    def apply(inputs):
        x = inputs
        if block_type == "X":
            x = XBlock(
                filters_in, filters_out, group_width, stride=2, norm='bn', name=f"{name}_XBlock_0"
            )(x)
            for i in range(1, depth):
                x = XBlock(
                    filters_out, filters_out, group_width, norm='bn', name=f"{name}_XBlock_{i}"
                )(x)
        elif block_type == "Y":
            x = YBlock(
                filters_in, filters_out, group_width, stride=2, name=name + "_YBlock_0"
            )(x)
            for i in range(1, depth):
                x = YBlock(
                    filters_out, filters_out, group_width, name=f"{name}_YBlock_{i}"
                )(x)
        elif block_type == "Z":
            x = ZBlock(
                filters_in, filters_out, group_width, stride=2, name=f"{name}_ZBlock_0"
            )(x)
            for i in range(1, depth):
                x = ZBlock(
                    filters_out, filters_out, group_width, name=f"{name}_ZBlock_{i}"
                )(x)
        else:
            raise NotImplementedError(
                f"Block type `{block_type}` not recognized."
                f"block_type must be one of (`X`, `Y`, `Z`). "
            )
        return x

    return apply


def Head(num_classes=1000, activation="softmax", name=None):
    """Implementation of classification head of RegNet.
    Args:
      num_classes: number of classes for Dense layer
      name: name prefix
    Returns:
      Output logits tensor.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    def apply(x):
        x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
        x = layers.Dense(num_classes, activation=activation, name=name + "head_dense")(
            x
        )
        return x

    return apply


def RegNet(
    depths,
    widths,
    group_width,
    block_type,
    input_shape=None,
    stem_strides=2,
    stem_filters=32,
    model_name="regnet",
    input_tensor=None,
    include_top=True,
    pooling=None,
    classes=1000,
    norm='bn',
    classifier_activation="softmax",
):
    """Instantiates RegNet architecture given specific configuration.
    Args:
      depths: An iterable containing depths for each individual stages.
      widths: An iterable containing output channel width of each individual
        stages
      group_width: Number of channels to be used in each group. See grouped
        convolutions for more information.
      block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
        papers "Designing network design spaces" and "Fast and Accurate Model
        Scaling"
      default_size: Default input image size.
      model_name: An optional name for the model.
      include_top: Boolean denoting whether to include classification head to the
        model.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use
        as image input for the model.
      input_shape: optional shape tuple, only to be specified if `include_top` is
        False. It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction when `include_top` is
        `False`. - `None` means that the output of the model will be the 4D tensor
        output of the last convolutional layer. - `avg` means that global average
        pooling will be applied to the output of the last convolutional layer, and
        thus the output of the model will be a 2D tensor. - `max` means that
        global max pooling will be applied.
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
        ValueError: if `classifier_activation` is not `softmax` or `None` when
          using a pretrained top layer.
        ValueError: if `include_top` is True but `num_classes` is not 1000.
        ValueError: if `block_type` is not one of `{"X", "Y", "Z"}`
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    x = inputs
    x = Stem(name=model_name, n_filters=stem_filters, norm=norm, strides=stem_strides)(x)

    # add identity layer to better reference the name later
    x = tf.keras.layers.Activation('linear', name=f'stem')(x)

    in_channels = stem_filters  # Output from Stem

    for num_stage in range(4):
        depth = depths[num_stage]
        out_channels = widths[num_stage]

        x = Stage(
            block_type,
            depth,
            group_width,
            in_channels,
            out_channels,
            norm=norm,
            name=model_name + "_Stage_" + str(num_stage),
        )(x)
        # add identity layer to better reference the name later
        x = tf.keras.layers.Activation('linear', name=f'stage_{num_stage}')(x)
        in_channels = out_channels

    if include_top:
        x = Head(num_classes=classes, activation=classifier_activation)(x)

    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D()(x)

    model = training.Model(inputs=inputs, outputs=x, name=model_name)
    return model


## Instantiating variants ##


def RegNetX002(
    model_name="regnetx002",
    include_top=True,
    input_tensor=None,
    input_shape=None,
    stem_strides=2,
    stem_filters=32,
    pooling=None,
    classes=1000,
    norm='bn',
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x002"]["depths"],
        MODEL_CONFIGS["x002"]["widths"],
        MODEL_CONFIGS["x002"]["group_width"],
        MODEL_CONFIGS["x002"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        input_tensor=input_tensor,
        input_shape=input_shape,
        stem_strides=stem_strides,
        stem_filters=stem_filters,
        pooling=pooling,
        classes=classes,
        norm=norm,
        classifier_activation=classifier_activation,
    )

def RegNetX001(
    model_name="regnetx001",
    include_top=True,
    input_tensor=None,
    input_shape=None,
    stem_strides=2,
    stem_filters=32,
    pooling=None,
    classes=1000,
    norm='bn',
    classifier_activation="softmax",
):
    return RegNet(
        MODEL_CONFIGS["x001"]["depths"],
        MODEL_CONFIGS["x001"]["widths"],
        MODEL_CONFIGS["x001"]["group_width"],
        MODEL_CONFIGS["x001"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        input_tensor=input_tensor,
        input_shape=input_shape,
        stem_strides=stem_strides,
        stem_filters=stem_filters,
        pooling=pooling,
        classes=classes,
        norm=norm,
        classifier_activation=classifier_activation,
    )


##### UNET 



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

    # backbone = RegNetX002(input_shape=(512,512,1), stem_strides=1, include_top=False)
    backbone = RegNetX001(input_shape=(512,512,1), stem_strides=1, norm='ln', include_top=False)

    backbone.summary() 

    unet = UnetDecoderRegnetx(classes=32, activation='linear')(backbone)

    model = tf.keras.Model(backbone.inputs, [unet])


    x = np.zeros((1,512,512,1))
    x[:,256,256] = 1
    x[:,::16] = 1
    u = model.predict(x)
