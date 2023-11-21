from mmseg.models import build_head as build_head_mmseg
from .deeplabv3plus import DeepLabV3PlusHead

def build_head(cfg):
    if cfg['decode_head'] == "DeepLabV3PlusHead":
        head = DeepLabV3PlusHead(
            dilations=cfg['dilations'],
            low_channels=cfg["low_channels"],
            high_channels=cfg["high_channels"],
            num_classes=cfg['nclass'])
    elif cfg['decode_head'] == "SegformerHead":
        head = build_head_mmseg(
            dict(
                type='SegformerHead',
                in_channels=cfg["in_channels"],
                in_index=[0, 1, 2, 3],
                channels=cfg["channels"],
                dropout_ratio=0.1,
                num_classes=cfg['nclass'],
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            )
        )
        head.init_weights()
    else:
        raise NotImplementedError(f"{cfg['decode_head']} not implemented!")
    return head