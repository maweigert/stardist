import numpy as np
import segmentation_models
import tensorflow as tf
from csbdeep.internals.blocks import unet_block
from csbdeep.utils.tf import keras_import
keras = keras_import()
K = keras_import('backend')
Input, Conv2D, MaxPooling2D, UpSampling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D', 'UpSampling2D')
Model = keras_import('models', 'Model')
from ._attunet import AttentionUnet

from .backbones._convnext import UnetConvNext 
from .backbones._unet import unet_block
from .backbones._msrf import msrf

def get_backbone2d(input_img, config):
    """ 
    return features prob, dist, probclass
    """
    def _pool_grid(x):
        grid = np.asarray(config.grid)
        assert all(g==grid[0] for g in grid)
        pooled = np.array([1,1])
        while tuple(pooled) != tuple(grid):
            pool = 1 + (grid > pooled)
            pooled *= pool
            for _ in range(config.unet_n_conv_per_depth):
                x = Conv2D(config.unet_n_filter_base, config.unet_kernel_size,
                                    padding='same', activation=config.unet_activation)(x)
            x = MaxPooling2D(pool)(x)
        return x 

    if config.backbone == "unet":
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(config).items() if k.startswith('unet_')}
        pooled_img = _pool_grid(input_img)
        backbone = unet_block(**unet_kwargs)(pooled_img)

    elif config.backbone == "unetv2":
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(config).items() if k.startswith('unet_')}
        unet_kwargs['expansion'] =  1.5
        global_pool = 2 
        pooled_img = Conv2D(config.unet_n_filter_base, 5 ,strides=(global_pool, global_pool),
                                padding='same', activation=config.unet_activation)(input_img)
        pooled_img = _pool_grid(pooled_img)
        backbone = unet_block(**unet_kwargs)(pooled_img)
        backbone = UpSampling2D(global_pool)(backbone)
        
    elif config.backbone == "seunet":
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(config).items() if k.startswith('unet_')}
        pooled_img = _pool_grid(input_img)
        backbone = unet_block(squeeze_excite = True, **unet_kwargs)(pooled_img)

    elif config.backbone == "seunetv2":
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(config).items() if k.startswith('unet_')}
        unet_kwargs['expansion'] =  1.5
        global_pool = 2 
        pooled_img = Conv2D(config.unet_n_filter_base, 5 ,strides=(global_pool, global_pool),
                                padding='same', activation=config.unet_activation)(input_img)
        pooled_img = _pool_grid(pooled_img)
        backbone = unet_block(squeeze_excite = True, **unet_kwargs)(pooled_img)
        backbone = UpSampling2D(global_pool)(backbone)

    elif config.backbone == "msrf":
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(config).items() if k.startswith('unet_')}
        global_pool = 2 
        pooled_img = Conv2D(config.unet_n_filter_base, 5 ,strides=(global_pool, global_pool),
                                padding='same', activation=config.unet_activation)(input_img)
        pooled_img = _pool_grid(pooled_img)
        backbone = msrf(input_size=(None,None,config.unet_n_filter_base))(pooled_img)
        backbone = UpSampling2D(global_pool)(backbone)
    

        # backbone = Conv2D(256, 3, padding='same', activation='linear')(backbone)
    elif config.backbone == "attunet":
        global_pool = 2 
        pooled_img = Conv2D(config.unet_n_filter_base, 5 ,strides=(global_pool, global_pool),
                                padding='same', activation=config.unet_activation)(input_img)
        pooled_img = _pool_grid(pooled_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = AttentionUnet(dim=config.unet_n_filter_base, 
                        in_channels = _inp_channel, out_channels=config.unet_n_filter_base,
                        dim_mults=(1, 2, 3, 4, 5), full_attention=False, use_convtranspose=False)(pooled_img)
        backbone = UpSampling2D(global_pool)(backbone)
        
    elif config.backbone == "regnetx":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        # dummy clall such that internal libs are imported 
        segmentation_models.Unet('resnet18', encoder_weights=None)
        base = tf.keras.applications.regnet.RegNetX002(include_top=False, include_preprocessing=False, weights=None, input_shape=(None,None,_inp_channel))
        features = ("regnetx002_Stage_0_XBlock_0_conv_1x1_1_relu", "regnetx002_Stage_1_XBlock_0_conv_1x1_1_relu", 
                    "regnetx002_Stage_2_XBlock_0_conv_1x1_1_relu", "regnetx002_Stage_3_XBlock_0_conv_1x1_1_relu")
        backbone = segmentation_models.models.unet.build_unet(base, 
                            decoder_block=segmentation_models.models.unet.DecoderUpsamplingX2Block, 
                            skip_connection_layers=features[::-1],
                            classes=128, activation='linear')(pooled_img)                
    elif config.backbone == "unet_efficientnet":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.Unet('efficientnetb0', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            activation='linear', classes=256, decoder_use_batchnorm=False)(pooled_img)        
    elif config.backbone == "fpn_resnet18":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.FPN('resnet18', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            pyramid_block_filters=128, classes=256, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_resnext50":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.FPN('resnext50', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            pyramid_block_filters=128, classes=256, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_seresnet18":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.FPN('seresnet18', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            pyramid_block_filters=128, classes=256, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_efficientnet":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.FPN('efficientnetb0', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            pyramid_block_filters=64, classes=128, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_densenet121":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        backbone = segmentation_models.FPN('densenet121', encoder_weights=None, input_shape=(None,None,_inp_channel), 
                            pyramid_block_filters=128, classes=256, activation="linear")(pooled_img)        
    elif config.backbone == "convnext":
        pooled_img = _pool_grid(input_img)
        _inp_channel = config.unet_n_filter_base if max(config.grid)>1 else 1
        # dummy clall such that internal libs are imported 
        backbone = UnetConvNext(input_shape=(None,None, _inp_channel), classes=128, activation='linear')(pooled_img)
    else: 
        raise KeyError(config.backbone)

    return backbone

if __name__ == "__main__":
    from stardist.models import Config2D, StarDist2D

    conf = Config2D(backbone='convnext', n_classes=1)
    model = StarDist2D(conf, None,None)

    model.keras_model.summary()

    x = np.zeros((1024,1024))
    x[512,512] = 1 

    # x = np.zeros((16,16))

    p,d, pc = model.predict(x)
