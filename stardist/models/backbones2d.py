import numpy as np
import segmentation_models
from csbdeep.internals.blocks import unet_block
from csbdeep.utils.tf import keras_import
keras = keras_import()
K = keras_import('backend')
Input, Conv2D, MaxPooling2D, UpSampling2D = keras_import('layers', 'Input', 'Conv2D', 'MaxPooling2D', 'UpSampling2D')
Model = keras_import('models', 'Model')

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
    elif config.backbone == "fpn_resnet18":
        pooled_img = _pool_grid(input_img)
        backbone = segmentation_models.FPN('resnet18', encoder_weights=None, input_shape=(None,None,config.unet_n_filter_base) if max(config.grid)>1 else (None,None,1), 
                            pyramid_block_filters=128, classes=128, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_resnext50":
        pooled_img = _pool_grid(input_img)
        backbone = segmentation_models.FPN('resnext50', encoder_weights=None, input_shape=(None,None,config.unet_n_filter_base) if max(config.grid)>1 else (None,None,1), 
                            pyramid_block_filters=128, classes=128, activation="linear")(pooled_img)        
    elif config.backbone == "fpn_seresnet18":
        pooled_img = _pool_grid(input_img)
        backbone = segmentation_models.FPN('seresnet18', encoder_weights=None, input_shape=(None,None,config.unet_n_filter_base) if max(config.grid)>1 else (None,None,1), 
                            pyramid_block_filters=128, classes=128, activation="linear")(pooled_img)        
    elif config.backbone == "linknet":
        backbone = segmentation_models.Linknet('resnet18', encoder_weights=None, input_shape=config.net_input_shape, classes=128, activation="linear")(input_img)
    else: 
        raise KeyError(config.backbone)

    return backbone

if __name__ == "__main__":
    from stardist.models import Config2D, StarDist2D

    # conf = Config2D(backbone='fpn_resnext50', n_classes=1)
    # conf = Config2D(backbone='fpn_resnet18', n_classes=1)
    conf = Config2D(backbone='fpn_seresnet18', n_classes=1)
    model = StarDist2D(conf, None,None)

    x = np.zeros((1024,1024))
    x[512,512] = 1 

    # x = np.zeros((16,16))

    p,d, pc = model.predict(x)
