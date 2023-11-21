import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml

from dataset.semi_dct import SemiDatasetDCT
from model.semseg.segmentor import Segmentor
from supervised_dct import evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.classes import CLASSES

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        print('{}\n'.format(pprint.pformat(cfg)))


    cudnn.enabled = True
    cudnn.benchmark = True

    model = Segmentor(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    # print(model)
    if rank == 0:
        print('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    valset = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'val')

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)

    if cfg['dataset'] == 'cityscapes':
        eval_mode = 'sliding_window'
    else:
        eval_mode = 'original'
    mIOU, iou_class = evaluate(model, valloader, eval_mode, cfg, local_rank)
    if rank == 0:
        print('***** Evaluation {} ***** >>>> meanIOU: {:.3f}\n'.format(eval_mode, mIOU))
        iou_class = [(cls_idx, iou) for cls_idx, iou in enumerate(iou_class)]
        iou_class.sort(key=lambda x:x[1])
        for (cls_idx, iou) in iou_class:
            print('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))

if __name__ == '__main__':
    main()
