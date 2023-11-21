
import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
import yaml

from dataset.semi_dct import SemiDatasetDCT
from dataset.semi import SemiDataset
from model.semseg.segmentor import Segmentor
from supervised import evaluate
from supervised_dct import evaluate as evaluate_dct
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--config2', type=str, required=True)
parser.add_argument('--config3', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--thr', default=0.95, type=float)
parser.add_argument('--uw', default=1.0, type=float)


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg2 = yaml.load(open(args.config2, "r"), Loader=yaml.Loader)
    cfg3 = yaml.load(open(args.config3, "r"), Loader=yaml.Loader)
    cfg['conf_thresh'] = args.thr
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, word_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': word_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model1 = Segmentor(cfg)
    model2 = Segmentor(cfg2)
    model3 = Segmentor(cfg3)
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model1)))

    param_groups1 = [{'params': model1.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model1.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    param_groups2 = [{'params': model2.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model2.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    param_groups3 = [{'params': model3.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model3.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}]
    if cfg["optim"] == "SGD":
        optimizer1 = SGD(param_groups1, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg["optim"] == "AdamW":
        optimizer1 = AdamW(param_groups1, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg2["optim"] == "SGD":
        optimizer2 = SGD(param_groups2, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg2["optim"] == "AdamW":
        optimizer2 = AdamW(param_groups2, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))
    if cfg3["optim"] == "SGD":
        optimizer3 = SGD(param_groups3, lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    elif cfg3["optim"] == "AdamW":
        optimizer3 = AdamW(param_groups3, lr=cfg['lr'], weight_decay=0.01, betas=(0.9, 0.999))

    local_rank = int(os.environ["LOCAL_RANK"])
    model1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model1).cuda()
    model2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model2).cuda()
    model3 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model3).cuda()

    model1 = torch.nn.parallel.DistributedDataParallel(model1, device_ids=[local_rank], 
                                                       output_device=local_rank, find_unused_parameters=False)
    model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[local_rank],
                                                       output_device=local_rank, find_unused_parameters=False)
    model3 = torch.nn.parallel.DistributedDataParallel(model3, device_ids=[local_rank],
                                                       output_device=local_rank, find_unused_parameters=False)
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valset_dct = SemiDatasetDCT(cfg['dataset'], cfg['data_root'], 'val')


    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                                pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler)
    valsampler_dct = torch.utils.data.distributed.DistributedSampler(valset_dct)
    valloader_dct = DataLoader(valset_dct, batch_size=1, pin_memory=True, num_workers=2,
                           drop_last=False, sampler=valsampler_dct)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best1, previous_best2, previous_best3 = 0.0, 0.0, 0.0

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.4f}, '
                        'Previous best1: {:.2f}, Previous best2: {:.2f}, Previous best3: {:.2f}'.format(
                epoch, optimizer1.param_groups[0]['lr'], previous_best1, previous_best2, previous_best3))

        total_loss_x1, total_loss_s1 = 0.0, 0.0
        total_mask_ratio1 = 0.0
        total_loss_x2, total_loss_s2 = 0.0, 0.0
        
        total_mask_ratio2 = 0.0

        total_agree_ratio_between3 = 0.0
        total_agree_ratio_between2 = 0.0

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((img_x_dct, img_x, mask_x, img_shape),
                (img_u_w, img_u_s, img_u_w_dct, img_u_s_dct, ignore_mask, _, cutmix_box_dct_img, cutmix_box, _, _)) in enumerate(loader):

            img_x_dct, img_x, mask_x = img_x_dct.cuda(), img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            img_u_w_dct, img_u_s_dct = img_u_w_dct.cuda(), img_u_s_dct.cuda()
            ignore_mask, cutmix_box = ignore_mask.cuda(), cutmix_box.cuda()
            cutmix_box_dct_img = cutmix_box_dct_img.cuda()
            
            index = torch.randperm(img_u_w.size(0))
            img_u_s_mix = img_u_s.clone()[index]
            img_u_s_dct_mix = img_u_s_dct.clone()[index]

            img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s_mix.shape) == 1]
            img_u_s_dct[cutmix_box_dct_img.unsqueeze(1).expand(img_u_s_dct.shape) == 1] = \
                img_u_s_dct_mix[cutmix_box_dct_img.unsqueeze(1).expand(img_u_s_dct_mix.shape) == 1]


            model1.train()
            model2.train()
            model3.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            pred_x1, pred_u_w1 = model1(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
            pred_u_s1 = model1(img_u_s)
            pred_x2, pred_u_w2 = model2(torch.cat((img_x_dct, img_u_w_dct)), img_shape[0]).split([num_lb, num_ulb])
            pred_u_s2 = model2(img_u_s_dct, img_shape[0])
            pred_x3, pred_u_w3 = model3(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])
            pred_u_s3 = model3(img_u_s)

            pred_u_w1 = pred_u_w1.detach() # bs, c, h, w
            conf_u_w1 = pred_u_w1.softmax(dim=1).max(dim=1)[0]
            mask_u_w1 = pred_u_w1.argmax(dim=1) # bs, h, w

            pred_u_w2 = pred_u_w2.detach() # bs, c, h, w
            conf_u_w2 = pred_u_w2.softmax(dim=1).max(dim=1)[0] # bs, h, w
            mask_u_w2 = pred_u_w2.argmax(dim=1) # bs, h, w
            
            pred_u_w3 = pred_u_w3.detach() # bs, c, h, w
            conf_u_w3 = pred_u_w3.softmax(dim=1).max(dim=1)[0] # bs, h, w
            mask_u_w3 = pred_u_w3.argmax(dim=1) # bs, h, w

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w1.clone(), conf_u_w1.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w2.clone(), conf_u_w2.clone()
            mask_u_w_cutmixed3, conf_u_w_cutmixed3 = mask_u_w3.clone(), conf_u_w3.clone()
            ignore_mask_cutmixed1, ignore_mask_cutmixed2, ignore_mask_cutmixed3 = \
                ignore_mask.clone(), ignore_mask.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box == 1] = mask_u_w1.clone()[index][cutmix_box == 1]
            conf_u_w_cutmixed1[cutmix_box == 1] = conf_u_w1.clone()[index][cutmix_box == 1]
            mask_u_w_cutmixed2[cutmix_box == 1] = mask_u_w2.clone()[index][cutmix_box == 1]
            conf_u_w_cutmixed2[cutmix_box == 1] = conf_u_w2.clone()[index][cutmix_box == 1]
            mask_u_w_cutmixed3[cutmix_box == 1] = mask_u_w3.clone()[index][cutmix_box == 1]
            conf_u_w_cutmixed3[cutmix_box == 1] = conf_u_w3.clone()[index][cutmix_box == 1]
            
            ignore_mask_cutmixed1[cutmix_box == 1] = ignore_mask.clone()[index][cutmix_box == 1]
            ignore_mask_cutmixed2[cutmix_box == 1] = ignore_mask.clone()[index][cutmix_box == 1]
            ignore_mask_cutmixed3[cutmix_box == 1] = ignore_mask.clone()[index][cutmix_box == 1]

            loss_x1 = criterion_l(pred_x1, mask_x)
            loss_x2 = criterion_l(pred_x2, mask_x)
            loss_x3 = criterion_l(pred_x3, mask_x)

            
            # supervise model1 with pred2 and pred3
            loss_u_s11 = criterion_u(pred_u_s1, mask_u_w_cutmixed2)
            loss_u_s11 = loss_u_s11 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s11 = torch.sum(loss_u_s11) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            
            loss_u_s12 = criterion_u(pred_u_s1, mask_u_w_cutmixed3)
            loss_u_s12 = loss_u_s12 * ((conf_u_w_cutmixed3 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s12 = torch.sum(loss_u_s12) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_s1 = (loss_u_s11 + loss_u_s12) / 2.0

            # supervise model2 with pred1 and pred3
            loss_u_s21 = criterion_u(pred_u_s2, mask_u_w_cutmixed1)
            loss_u_s21 = loss_u_s21 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s21 = torch.sum(loss_u_s21) / torch.sum(ignore_mask_cutmixed2 != 255).item()
            
            loss_u_s22 = criterion_u(pred_u_s2, mask_u_w_cutmixed3)
            loss_u_s22 = loss_u_s22 * ((conf_u_w_cutmixed3 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s22 = torch.sum(loss_u_s22) / torch.sum(ignore_mask_cutmixed2 != 255).item()
            loss_u_s2 = (loss_u_s21 + loss_u_s22) / 2.0
            
            # supervise model3 with pred1 and pred2
            loss_u_s31 = criterion_u(pred_u_s3, mask_u_w_cutmixed1)
            loss_u_s31 = loss_u_s31 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed3 != 255))
            loss_u_s31 = torch.sum(loss_u_s31) / torch.sum(ignore_mask_cutmixed3 != 255).item()
            
            loss_u_s32 = criterion_u(pred_u_s3, mask_u_w_cutmixed2)
            loss_u_s32 = loss_u_s32 * ((mask_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed3 != 255))
            loss_u_s32 = torch.sum(loss_u_s32) / torch.sum(ignore_mask_cutmixed3 != 255).item()
            loss_u_s3 = (loss_u_s31 + loss_u_s32) / 2.0
            
            loss = (loss_x1 + loss_x2 + loss_x3) / 3.0 + args.uw * ((loss_u_s1 + loss_u_s2 + loss_u_s3) / 3.0)

            torch.distributed.barrier()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            total_loss_x1 += loss_x1.item()
            total_loss_s1 += loss_u_s1.item()
            total_mask_ratio1 += ((conf_u_w1 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                                 (ignore_mask != 255).sum().item()

            total_loss_x2 += loss_x2.item()
            total_loss_s2 += loss_u_s2.item()
            total_mask_ratio2 += ((conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                                 (ignore_mask != 255).sum().item()

            total_agree_ratio_between3 += \
                (((mask_u_w2 == mask_u_w1) & (mask_u_w2 == mask_u_w3)) * ((conf_u_w1 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (conf_u_w3 >= cfg['conf_thresh']) & (ignore_mask != 255))).sum().item() / \
                ((conf_u_w1 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item()
            
            
            agree_ratio12 = ((mask_u_w2 == mask_u_w1) * ((conf_u_w1 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255))).sum().item() / \
                ((conf_u_w1 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item()
            agree_ratio23 = ((mask_u_w2 == mask_u_w3) * ((conf_u_w3 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255))).sum().item() / \
                ((conf_u_w3 >= cfg['conf_thresh']) & (conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item()
            agree_ratio13 = ((mask_u_w1 == mask_u_w3) * ((conf_u_w3 >= cfg['conf_thresh']) & (conf_u_w1 >= cfg['conf_thresh']) & (ignore_mask != 255))).sum().item() / \
                ((conf_u_w3 >= cfg['conf_thresh']) & (conf_u_w1 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item()
            
            total_agree_ratio_between2 += (agree_ratio12 + agree_ratio13 + agree_ratio23) / 3.0
                
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer1.param_groups[1]["lr"] = lr * cfg['lr_multi']
            optimizer2.param_groups[0]["lr"] = lr
            optimizer2.param_groups[1]["lr"] = lr * cfg['lr_multi']
            optimizer3.param_groups[0]["lr"] = lr
            optimizer3.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if (i % 50 == 0) and (rank == 0):
                logger.info('Iters: {:}, Loss x1: {:.3f}, Loss s1: {:.3f}, Mask1: {:.3f}, '
                            'Loss x2: {:.3f}, Loss s2: {:.3f}, Mask2: {:.3f}, agree ratio (between 3): {:.3f} agree ratio (between 2): {:.3f}'.format(
                    i, total_loss_x1 / (i+1), total_loss_s1 / (i+1), total_mask_ratio1 / (i+1), 
                    total_loss_x2 / (i+1), total_loss_s2 / (i+1), total_mask_ratio2 / (i+1), total_agree_ratio_between3 / (i+1), total_agree_ratio_between2 / (i+1)))

        torch.cuda.empty_cache()

        if cfg['dataset'] == 'cityscapes':
            # eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
            eval_mode = 'sliding_window'
        else:
            eval_mode = 'original'
        mIOU1, iou_class1 = evaluate(model1, valloader, eval_mode, cfg)
        mIOU2, iou_class2 = evaluate_dct(model2, valloader_dct, eval_mode, cfg2)
        mIOU3, iou_class3 = evaluate(model3, valloader, eval_mode, cfg3)

        if rank == 0:
            # iou_class1 = [(cls_idx, iou) for cls_idx, iou in enumerate(iou_class1)]
            # iou_class1.sort(key=lambda x:x[1])
            for (cls_idx, iou) in enumerate(iou_class1):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU1: {:.2f}, '
                            'IoU2: {:.2f}, IoU3: {:.2f},'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class2[cls_idx], iou_class3[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU1: {:.2f}, MeanIoU2: {:.2f}, MeanIoU3: {:.2f}\n'.format(eval_mode, mIOU1, mIOU2, mIOU3))

        if mIOU1 > previous_best1 and rank == 0:
            if previous_best1 != 0:
                pre_path = os.path.join(args.save_path, 'm1_%s_%.2f.pth' % (cfg['backbone'], previous_best1))
                if os.path.exists(pre_path):
                    os.remove(pre_path)
            previous_best1 = mIOU1
            torch.save(model1.module.state_dict(),
                       os.path.join(args.save_path, 'm1_%s_%.2f.pth' % (cfg['backbone'], mIOU1)))
        
        if mIOU2 > previous_best2 and rank == 0:
            if previous_best2 != 0:
                pre_path = os.path.join(args.save_path, 'm2_%s_%.2f.pth' % (cfg2['backbone'], previous_best2))
                if os.path.exists(pre_path):
                    os.remove(pre_path)
            previous_best2 = mIOU2
            torch.save(model2.module.state_dict(),
                       os.path.join(args.save_path, 'm2_%s_%.2f.pth' % (cfg2['backbone'], mIOU2)))
        
        if mIOU3 > previous_best3 and rank == 0:
            if previous_best3 != 0:
                pre_path = os.path.join(args.save_path, 'm3_%s_%.2f.pth' % (cfg3['backbone'], previous_best3))
                if os.path.exists(pre_path):
                    os.remove(pre_path)
            previous_best3 = mIOU3
            torch.save(model3.module.state_dict(),
                       os.path.join(args.save_path, 'm3_%s_%.2f.pth' % (cfg3['backbone'], mIOU3)))
        # previous_best1 = max(previous_best1, mIOU1)
        # previous_best2 = max(previous_best2, mIOU2)

if __name__ == '__main__':
    main()