# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .inline_swin import InLineSwin
from .inline_deit import inline_deit_tiny, inline_deit_small, inline_deit_base
from .inline_pvt import inline_pvt_tiny, inline_pvt_small, inline_pvt_medium, inline_pvt_large
from .inline_cswin import inline_cswin_tiny, inline_cswin_small, inline_cswin_base, inline_cswin_base_384


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'inline_swin':
        model = InLineSwin(img_size=config.DATA.IMG_SIZE,
                           patch_size=config.MODEL.SWIN.PATCH_SIZE,
                           in_chans=config.MODEL.SWIN.IN_CHANS,
                           num_classes=config.MODEL.NUM_CLASSES,
                           embed_dim=config.MODEL.SWIN.EMBED_DIM,
                           depths=config.MODEL.SWIN.DEPTHS,
                           num_heads=config.MODEL.SWIN.NUM_HEADS,
                           window_size=config.MODEL.SWIN.WINDOW_SIZE,
                           mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                           qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                           qk_scale=config.MODEL.SWIN.QK_SCALE,
                           drop_rate=config.MODEL.DROP_RATE,
                           drop_path_rate=config.MODEL.DROP_PATH_RATE,
                           ape=config.MODEL.SWIN.APE,
                           patch_norm=config.MODEL.SWIN.PATCH_NORM,
                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                           attn_type=config.MODEL.INLINE.ATTN_TYPE)

    elif model_type in ['inline_deit_tiny', 'inline_deit_small', 'inline_deit_base']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE)')

    elif model_type in ['inline_pvt_tiny', 'inline_pvt_small', 'inline_pvt_medium', 'inline_pvt_large']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.INLINE.ATTN_TYPE,'
                                  'la_sr_ratios=str(config.MODEL.INLINE.PVT_LA_SR_RATIOS))')

    elif model_type in ['inline_cswin_tiny', 'inline_cswin_small', 'inline_cswin_base', 'inline_cswin_base_384']:
        model = eval(model_type + '(img_size=config.DATA.IMG_SIZE,'
                                  'in_chans=config.MODEL.SWIN.IN_CHANS,'
                                  'num_classes=config.MODEL.NUM_CLASSES,'
                                  'drop_rate=config.MODEL.DROP_RATE,'
                                  'drop_path_rate=config.MODEL.DROP_PATH_RATE,'
                                  'attn_type=config.MODEL.INLINE.ATTN_TYPE,'
                                  'la_split_size=config.MODEL.INLINE.CSWIN_LA_SPLIT_SIZE)')

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
