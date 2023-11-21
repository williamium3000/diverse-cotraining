from turbojpeg import TurboJPEG
import numpy as np
from jpeg2dct.numpy import load, loads
import cv2
import torch
from .dct_subset import *
from .dct_norm_statistics import *
import torchvision.transforms.functional as F

libjpeg_path = 'libjpeg-turbo-2.0.3/build/libturbojpeg.so'

jpeg_encoder = TurboJPEG(libjpeg_path)

class DCTTransform():
    def __init__(self, dct_channels=64, pattern="square"):
        self.subset_dct = SubsetDCT(dct_channels, pattern)
        self.norm_dct = NormalizeDCT(
            train_upscaled_static_mean,
            train_upscaled_static_std,
            channels=dct_channels,
            pattern=pattern
        )
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = upscale(img, upscale_factor=2, interpolation=cv2.INTER_LINEAR)
        img_up = upscale(img, upscale_factor=2, interpolation=cv2.INTER_LINEAR)
        y, cb, cr = transform_upscaled_dct(img, img_up)
        y, cb, cr = to_tensor_dct(y, cb, cr)
        y, cb, cr = self.subset_dct([y, cb, cr])
        img = dct_aggregate(y, cb, cr)
        img, _, _ = self.norm_dct(img)
        return img


def upscale(img, upscale_factor=None, desize_size=None, interpolation=cv2.INTER_LINEAR):
    
    h, w, c = img.shape
    if upscale_factor is not None:
        dh, dw = upscale_factor*h, upscale_factor*w
    elif desize_size is not None:
        # dh, dw = desize_size.shape
        dh, dw = desize_size
    else:
        raise ValueError
    return cv2.resize(img, dsize=(dw, dh), interpolation=interpolation)


def transform_dct(img, encoder):
    if img.dtype != 'uint8':
        img = np.ascontiguousarray(img, dtype="uint8")
    img = encoder.encode(img, quality=100, jpeg_subsample=2)
    dct_y, dct_cb, dct_cr = loads(img)  # 28
    return dct_y, dct_cb, dct_cr

def transform_upscaled_dct(img, img_upscaled):
    y, cbcr = img, img_upscaled
    dct_y, _, _ = transform_dct(y, jpeg_encoder)
    _, dct_cb, dct_cr = transform_dct(cbcr, jpeg_encoder)
    return dct_y, dct_cb, dct_cr

def to_tensor_dct(y, cb, cr):
    y = torch.from_numpy(y.transpose((2, 0, 1))).float()
    cb = torch.from_numpy(cb.transpose((2, 0, 1))).float()
    cr = torch.from_numpy(cr.transpose((2, 0, 1))).float()
    return y, cb, cr


class SubsetDCT(object):
    def __init__(self, channels=64, pattern='square'):
        self.channels = channels

        if pattern == 'square':
            self.subset_channel_index = subset_channel_index_square
        elif pattern == 'learned':
            self.subset_channel_index = subset_channel_index_learned
        elif pattern == 'triangle':
            self.subset_channel_index = subset_channel_index_triangle

        if self.channels < 192:
            self.subset_y =  self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cr = self.subset_channel_index[channels][2]

    def __call__(self, tensor):
        if self.channels < 192:
            dct_y, dct_cb, dct_cr = tensor[0], tensor[1], tensor[2]
            dct_y, dct_cb, dct_cr = dct_y[self.subset_y], dct_cb[self.subset_cb], dct_cr[self.subset_cr]
            return dct_y, dct_cb, dct_cr
        else:
            return tensor[0], tensor[1], tensor[2]


def dct_aggregate(dct_y, dct_cb, dct_cr):
    return torch.cat((dct_y, dct_cb, dct_cr), dim=0)

class NormalizeDCT(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, y_mean, y_std, cb_mean=None, cb_std=None, cr_mean=None, cr_std=None, channels=None, pattern='square'):
        self.y_mean,  self.y_std = y_mean, y_std
        self.cb_mean, self.cb_std = cb_mean, cb_std
        self.cr_mean, self.cr_std = cr_mean, cr_std

        if channels < 192:
            if pattern == 'square':
                self.subset_channel_index = subset_channel_index_square
            elif pattern == 'learned':
                self.subset_channel_index = subset_channel_index_learned
            elif pattern == 'triangle':
                self.subset_channel_index = subset_channel_index_triangle

            self.subset_y  = self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cb = [64 + c for c in self.subset_cb]
            self.subset_cr = self.subset_channel_index[channels][2]
            self.subset_cr = [128 + c for c in self.subset_cr]
            self.subset = self.subset_y + self.subset_cb + self.subset_cr
            self.mean_y, self.std_y = [y_mean[i] for i in self.subset], [y_std[i] for i in self.subset]
        else:
            self.mean_y, self.std_y = y_mean, y_std


    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y  = F.normalize(y,  self.y_mean,  self.y_std)
            cb = F.normalize(cb, self.cb_mean, self.cb_std)
            cr = F.normalize(cr, self.cr_mean, self.cr_std)
            return y, cb, cr
        else:
            y = F.normalize(tensor, self.mean_y, self.std_y)
            return y, None, None
