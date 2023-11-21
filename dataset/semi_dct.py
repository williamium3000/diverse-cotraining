from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from .dct_transform import *
from .dct_norm_statistics import *

class SemiDatasetDCT(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None, dct_channels=64, pattern="square"):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open('partitions/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        self.dct_transform = DCTTransform(dct_channels=dct_channels, pattern=pattern)
        
    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))
        if self.mode == 'val':
            w, h = img.size
            img_shape = torch.tensor([h, w])
            img = self.dct_transform(img)
            mask = torch.from_numpy(np.array(mask)).long()
            return img, mask, id, img_shape

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        w, h = img.size
        img_shape = torch.tensor([h, w])
        if self.mode == 'train_l':
            img_dct = self.dct_transform(deepcopy(img))
            mask = torch.from_numpy(np.array(mask)).long()
            return img_dct, normalize(img), mask, img_shape

        img_w1, img_w2, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))
        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255
        
        img_w2 = self.dct_transform(img_w2)
        img_s2 = self.dct_transform(img_s2)
        cutmix_box2_img = obtain_cutmix_box(img_s2.shape[1], p=0.5)
        cutmix_box2_mask = interpolate(cutmix_box2_img.unsqueeze(0).unsqueeze(0), size=(mask.size(0), mask.size(1)), mode="nearest").squeeze(0).squeeze(0)
        
        return normalize(img_w1), img_s1, img_w2, img_s2, ignore_mask, cutmix_box1, cutmix_box2_img, cutmix_box2_mask, mask, img_shape

    def __len__(self):
        return len(self.ids)
