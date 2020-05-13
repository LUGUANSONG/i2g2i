#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import os
from os.path import exists, join
import json
import math
from collections import defaultdict
import random
from tqdm import tqdm
from copy import deepcopy
from torchvision import transforms

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel

from sg2im.data import imagenet_deprocess_batch
from sg2im.data.coco import CocoSceneGraphDataset, coco_collate_fn
from sg2im.data.vg import VgSceneGraphDataset, vg_collate_fn
from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.model import Sg2ImModel
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

from combine_sg2im_neural_motifs import load_detector

torch.backends.cudnn.benchmark = True


def check_args(args):
  H, W = args.image_size
  for _ in args.refinement_network_dims[1:]:
    H = H // 2
  if H == 0:
    raise ValueError("Too many layers in refinement network")


# def build_model(args): #, vocab):
#   if args.checkpoint_start_from is not None:
#     checkpoint = torch.load(args.checkpoint_start_from)
#     kwargs = checkpoint['model_kwargs']
#     model = Sg2ImModel(**kwargs)
#     raw_state_dict = checkpoint['model_state']
#     state_dict = {}
#     for k, v in raw_state_dict.items():
#       if k.startswith('module.'):
#         k = k[7:]
#       state_dict[k] = v
#     model.load_state_dict(state_dict)
#   else:
#     kwargs = {
#       # 'vocab': vocab,
#       'image_size': args.image_size,
#       'embedding_dim': args.embedding_dim,
#       'gconv_dim': args.gconv_dim,
#       'gconv_hidden_dim': args.gconv_hidden_dim,
#       'gconv_num_layers': args.gconv_num_layers,
#       'mlp_normalization': args.mlp_normalization,
#       'refinement_dims': args.refinement_network_dims,
#       'normalization': args.normalization,
#       'activation': args.activation,
#       'mask_size': args.mask_size,
#       'layout_noise_dim': args.layout_noise_dim,
#     }
#     model = Sg2ImModel(**kwargs)
#   return model, kwargs


def build_model(args, checkpoint):
  kwargs = checkpoint['model_kwargs']
  model = Sg2ImModel(**kwargs)
  raw_state_dict = checkpoint['model_state']
  state_dict = {}
  for k, v in raw_state_dict.items():
    if k.startswith('module.'):
      k = k[7:]
    state_dict[k] = v
  model.load_state_dict(state_dict)
  return model, kwargs


def check_model(args, loader, model, output_path):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  if not exists(output_path):
    os.makedirs(output_path)

  num_samples = 0
  img_cnt = 0
  with torch.no_grad():
    for batch in loader:
      imgs = F.interpolate(batch.imgs, size=args.image_size).to(args.sg2im_device)
      if args.num_gpus > 2:
        result = load_detector.detector.__getitem__(batch, target_device=args.detector_gather_device)
      else:
        result = load_detector.detector[batch]
      objs = result.obj_preds
      boxes = result.rm_box_priors
      obj_to_img = result.im_inds
      obj_fmap = result.obj_fmap
      if args.num_gpus == 2:
        objs = objs.to(args.sg2im_device)
        boxes = boxes.to(args.sg2im_device)
        obj_to_img = obj_to_img.to(args.sg2im_device)
        obj_fmap = obj_fmap.to(args.sg2im_device)
      boxes = boxes / load_detector.IM_SCALE

      # check if all image have detection
      cnt = torch.zeros(len(imgs)).byte()
      cnt[obj_to_img] += 1
      if (cnt > 0).sum() != len(imgs):
        print("some imgs have no detection")
        # print(obj_to_img)
        print(cnt)
        imgs = imgs[cnt]
        obj_to_img_new = obj_to_img.clone()
        for i in range(len(cnt)):
          if cnt[i] == 0:
            obj_to_img_new -= (obj_to_img > i).long()
        obj_to_img = obj_to_img_new

      model_out = model(obj_to_img, boxes, obj_fmap)
      imgs_pred = model_out

      # imgs = imagenet_deprocess_batch(imgs)
      # imgs_pred = imagenet_deprocess_batch(imgs_pred)
      imgs = imgs.cpu()
      imgs_pred = imgs_pred.cpu()
      images = imgs * torch.tensor([0.229, 0.224, 0.225], device=imgs.device).reshape(1, 3, 1, 1)
      images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
      # images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
      # images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
      # images = images - images_min
      # images = images / (images_max - images_min)
      imgs = images.clamp(min=0, max=1)

      images = imgs_pred * torch.tensor([0.229, 0.224, 0.225], device=imgs_pred.device).reshape(1, 3, 1, 1)
      images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
      # images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
      # images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
      # images = images - images_min
      # images = images / (images_max - images_min)
      imgs_pred = images.clamp(min=0, max=1)

      for img, img_pred in zip(imgs, imgs_pred):
        img = transforms.ToPILImage()(img).convert("RGB")
        img_pred = transforms.ToPILImage()(img_pred).convert("RGB")
        img.save(join(output_path, "img_%d.png" % img_cnt))
        img_pred.save(join(output_path, "img_pred_%d.png" % img_cnt))
        img_cnt += 1

      print("num_samples:", num_samples)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
        break


def main(args):
  print(args)
  check_args(args)
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  detector_gather_device = args.num_gpus - 1
  sg2im_device = torch.device(args.num_gpus - 1)
  args.detector_gather_device = detector_gather_device
  args.sg2im_device = sg2im_device

  vocab = {
    'object_idx_to_name': load_detector.train.ind_to_classes,
  }

  restore_path = '%s_with_model.pt' % args.checkpoint_name
  restore_path = os.path.join(args.output_dir, restore_path)
  print('Restoring from checkpoint:')
  print(restore_path)
  checkpoint = torch.load(restore_path, map_location=sg2im_device)

  model, model_kwargs = build_model(args, checkpoint)
  model = model.to(sg2im_device)
  print(model)

  model.eval()
  if True:
    print('checking on train')
    train_results = check_model(args, load_detector.train_loader, model, join(args.output_dir, "train"))
    print('checking on val')
    val_results = check_model(args, load_detector.val_loader, model, join(args.output_dir, "val"))

if __name__ == '__main__':
  args = load_detector.conf
  main(args)

