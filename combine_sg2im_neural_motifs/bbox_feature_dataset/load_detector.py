"""
Training script 4 Detection
"""
from mscoco import CocoDetection, CocoDataLoader
from visual_genome import VGDataLoader, VG
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

detector_num_gpus = conf.num_gpus
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
    train_not_flip, train_flip, val, test = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                                                      filter_empty_rels=False, use_proposals=conf.use_proposals)
    print("length of train_not_flip: %d" % len(train_not_flip))
    print("length of train_flip: %d" % len(train_flip))
    print("length of val: %d" % len(val))
    print("length of test: %d" % len(test))
    train_not_flip_loader, train_flip_loader, val_loader, test_loader = VGDataLoader.splits(
        train_not_flip, train_flip, val, test, batch_size=conf.batch_size,
        num_workers=conf.num_workers, num_gpus=detector_num_gpus)

detector = ObjectDetector(classes=train_not_flip.ind_to_classes, num_gpus=detector_num_gpus,
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
