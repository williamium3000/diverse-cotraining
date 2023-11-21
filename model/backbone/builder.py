import model.backbone.resnet_dct as resnet_dct
import model.backbone.resnet as resnet
from mmseg.models import build_backbone as build_backbone_mmseg
from .segformer import backbone_cfg as mit_cfg

def build_backbone(cfg):
    if 'dct' in cfg['backbone']:
        backbone = resnet_dct.__dict__[cfg['backbone']](True, multi_grid=cfg['multi_grid'],
                            replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
    elif 'resnet' in cfg['backbone']:
        backbone = resnet.__dict__[cfg['backbone']](True, multi_grid=cfg['multi_grid'],
                            replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
    elif "mit" in cfg['backbone']:
        backbone = build_backbone_mmseg(mit_cfg[cfg['backbone']])
        backbone.init_weights()
    return backbone