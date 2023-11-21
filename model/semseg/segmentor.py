from torch import nn
import torch.nn.functional as F
from model.backbone.builder import build_backbone
from model.head.builder import build_head

class Segmentor(nn.Module):
    def __init__(self, cfg):
        super(Segmentor, self).__init__()
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)

    def forward(self, x, img_shape=None):
        h, w = x.shape[-2:] if img_shape is None else img_shape

        feats = self.backbone(x)
        out = self.head(feats)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out


