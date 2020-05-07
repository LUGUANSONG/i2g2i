"""
Training script 4 Detection
"""
from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
from PIL import Image, ImageDraw, ImageFont


cudnn.benchmark = True
conf = ModelConfig()

if conf.num_gpus > 1:
    detector_num_gpus = conf.num_gpus - 1

if conf.coco:
    train, val = CocoDetection.splits()
    val.ids = val.ids[:conf.val_size]
    train.ids = train.ids
    train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     num_gpus=detector_num_gpus)
else:
    train, val, _ = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                              filter_empty_rels=False, use_proposals=conf.use_proposals)
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=detector_num_gpus)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=detector_num_gpus,
                          mode='refinerels' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet)
# print(detector)
# os._exit(0)
detector.cuda()

# Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers
if conf.use_proposals:
    for n, param in detector.named_parameters():
        if n.startswith('features'):
            param.requires_grad = False

start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)
    if optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = ckpt['epoch']
detector.eval()

#
# def train_epoch(epoch_num):
#     # detector.train()
#     detector.eval()
#     tr = []
#     start = time.time()
#     for b, batch in enumerate(train_loader):
#         tr.append(train_batch(batch))
#
#         if b % conf.print_interval == 0 and b >= conf.print_interval:
#             mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
#             time_per_batch = (time.time() - start) / conf.print_interval
#             print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
#                 epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
#             print(mn)
#             print('-----------', flush=True)
#             start = time.time()
#     return pd.concat(tr, axis=1)
#
#
# def train_batch(b):
#     '''
#     return Result(
#             od_obj_dists=od_obj_dists, # 1
#             rm_obj_dists=obj_dists, # 2
#             obj_scores=nms_scores, # 3
#             obj_preds=nms_preds, # 4
#             obj_fmap=obj_fmap, # 5 pick
#             od_box_deltas=od_box_deltas, # 6
#             rm_box_deltas=box_deltas, # 7
#             od_box_targets=bbox_targets, # 8
#             rm_box_targets=bbox_targets, # 9
#             od_box_priors=od_box_priors, # 10
#             rm_box_priors=box_priors, # 11 pick
#             boxes_assigned=nms_boxes_assign, # 12
#             boxes_all=nms_boxes, # 13
#             od_obj_labels=obj_labels, # 14
#             rm_obj_labels=rm_obj_labels, # 15
#             rpn_scores=rpn_scores, # 16
#             rpn_box_deltas=rpn_box_deltas, # 17
#             rel_labels=rel_labels, # 18
#             im_inds=im_inds, # 19 pick
#             fmap=fmap if return_fmap else None, # 20
#         )
#     '''
#     result = detector[b]
#     print("imgs.shape", b.imgs.shape)
#     print("im_sizes", b.im_sizes)
#     print("boxes", result.rm_box_priors)
#     print("im_inds", result.im_inds)
#     print("rm_obj_dists.shape", result.rm_obj_dists.shape)
#
#     for i in range(len(b.imgs)):
#         img_tensor = b.imgs[i].data.cpu()
#         print(img_tensor.shape, img_tensor.max(), img_tensor.min())
#         img_tensor = Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])(img_tensor)
#         img_tensor = Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])(img_tensor)
#         pil_img = ToPILImage()(img_tensor)
#         pil_img = pil_img.convert("RGB")
#         draw = ImageDraw.Draw(pil_img)
#         for j in range(len(result.rm_box_priors)):
#             if result.im_inds.data[j] == i:
#                 class_ind = int(result.rm_obj_dists.data[j].max(0)[1])
#                 if class_ind != 0:
#                     draw = draw_box(draw, result.rm_box_priors.data[j], train.ind_to_classes[class_ind])
#         pil_img.save("/newNAS/Workspaces/UCGroup/gslu/aws_ailab/code/neural-motifs/checkpoints/%d.png" % i)
#
#     return res
#
#
# print("Training starts now!")
# for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
#     rez = train_epoch(epoch)
#     print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
