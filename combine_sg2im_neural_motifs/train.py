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

from combine_sg2im_neural_motifs import train_detector

torch.backends.cudnn.benchmark = True


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def check_args(args):
    H, W = args.image_size
    for _ in args.refinement_network_dims[1:]:
        H = H // 2
    if H == 0:
        raise ValueError("Too many layers in refinement network")


def build_model(args): #, vocab):
    if args.checkpoint_start_from is not None:
        checkpoint = torch.load(args.checkpoint_start_from)
        kwargs = checkpoint['model_kwargs']
        model = Sg2ImModel(**kwargs)
        raw_state_dict = checkpoint['model_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        kwargs = {
            # 'vocab': vocab,
            'image_size': args.image_size,
            'embedding_dim': args.embedding_dim,
            'gconv_dim': args.gconv_dim,
            'gconv_hidden_dim': args.gconv_hidden_dim,
            'gconv_num_layers': args.gconv_num_layers,
            'mlp_normalization': args.mlp_normalization,
            'refinement_dims': args.refinement_network_dims,
            'normalization': args.normalization,
            'activation': args.activation,
            'mask_size': args.mask_size,
            'layout_noise_dim': args.layout_noise_dim,
        }
        model = Sg2ImModel(**kwargs)
    return model, kwargs


def build_obj_discriminator(args, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_obj_weight = args.d_obj_weight
    if d_weight == 0 or d_obj_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'vocab': vocab,
        'arch': args.d_obj_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'object_size': args.crop_size,
    }
    discriminator = AcCropDiscriminator(**d_kwargs)
    return discriminator, d_kwargs


def build_img_discriminator(args): #, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_img_weight = args.d_img_weight
    if d_weight == 0 or d_img_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'arch': args.d_img_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
    }
    discriminator = PatchDiscriminator(**d_kwargs)
    return discriminator, d_kwargs


# def build_coco_dsets(args):
#   dset_kwargs = {
#     'image_dir': args.coco_train_image_dir,
#     'instances_json': args.coco_train_instances_json,
#     'stuff_json': args.coco_train_stuff_json,
#     'stuff_only': args.coco_stuff_only,
#     'image_size': args.image_size,
#     'mask_size': args.mask_size,
#     'max_samples': args.num_train_samples,
#     'min_object_size': args.min_object_size,
#     'min_objects_per_image': args.min_objects_per_image,
#     'instance_whitelist': args.instance_whitelist,
#     'stuff_whitelist': args.stuff_whitelist,
#     'include_other': args.coco_include_other,
#     'include_relationships': args.include_relationships,
#   }
#   train_dset = CocoSceneGraphDataset(**dset_kwargs)
#   num_objs = train_dset.total_objects()
#   num_imgs = len(train_dset)
#   print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
#   print('(%.2f objects per image)' % (float(num_objs) / num_imgs))
#
#   dset_kwargs['image_dir'] = args.coco_val_image_dir
#   dset_kwargs['instances_json'] = args.coco_val_instances_json
#   dset_kwargs['stuff_json'] = args.coco_val_stuff_json
#   dset_kwargs['max_samples'] = args.num_val_samples
#   val_dset = CocoSceneGraphDataset(**dset_kwargs)
#
#   assert train_dset.vocab == val_dset.vocab
#   vocab = json.loads(json.dumps(train_dset.vocab))
#
#   return vocab, train_dset, val_dset


# def build_vg_dsets(args):
#   with open(args.vocab_json, 'r') as f:
#     vocab = json.load(f)
#   dset_kwargs = {
#     'vocab': vocab,
#     'h5_path': args.train_h5,
#     'image_dir': args.vg_image_dir,
#     'image_size': args.image_size,
#     'max_samples': args.num_train_samples,
#     'max_objects': args.max_objects_per_image,
#     'use_orphaned_objects': args.vg_use_orphaned_objects,
#     'include_relationships': args.include_relationships,
#   }
#   train_dset = VgSceneGraphDataset(**dset_kwargs)
#   iter_per_epoch = len(train_dset) // args.batch_size
#   print('There are %d iterations per epoch' % iter_per_epoch)
#
#   dset_kwargs['h5_path'] = args.val_h5
#   del dset_kwargs['max_samples']
#   val_dset = VgSceneGraphDataset(**dset_kwargs)
#
#   return vocab, train_dset, val_dset


# def build_loaders(args):
#   if args.dataset == 'vg':
#     vocab, train_dset, val_dset = build_vg_dsets(args)
#     collate_fn = vg_collate_fn
#   elif args.dataset == 'coco':
#     vocab, train_dset, val_dset = build_coco_dsets(args)
#     collate_fn = coco_collate_fn
#
#   loader_kwargs = {
#     'batch_size': args.batch_size,
#     'num_workers': args.loader_num_workers,
#     'shuffle': True,
#     'collate_fn': collate_fn,
#   }
#   train_loader = DataLoader(train_dset, **loader_kwargs)
#
#   loader_kwargs['shuffle'] = args.shuffle_val
#   val_loader = DataLoader(val_dset, **loader_kwargs)
#   return vocab, train_loader, val_loader


def check_model(args, t, loader, model):
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    num_samples = 0
    all_losses = defaultdict(list)
    # total_iou = 0
    # total_boxes = 0
    with torch.no_grad():
        for batch in loader:
        # for step, batch in enumerate(tqdm(loader, desc='Eval', total=len(loader))):
            # batch = [tensor.cuda() for tensor in batch]
            # masks = None
            # if len(batch) == 6:
            #   imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            # elif len(batch) == 7:
            #   imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
            # predicates = triples[:, 1]
            #
            # # Run the model as it has been run during training
            # model_masks = masks
            # model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
            # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

            imgs = F.interpolate(batch.imgs, size=args.image_size).to(args.sg2im_device)
            if args.num_gpus > 2:
                result = train_detector.detector.__getitem__(batch, target_device=args.detector_gather_device)
            else:
                result = train_detector.detector[batch]
            objs = result.obj_preds
            boxes = result.rm_box_priors
            obj_to_img = result.im_inds
            obj_fmap = result.obj_fmap
            if args.num_gpus == 2:
                objs = objs.to(args.sg2im_device)
                boxes = boxes.to(args.sg2im_device)
                obj_to_img = obj_to_img.to(args.sg2im_device)
                obj_fmap = obj_fmap.to(args.sg2im_device)
            boxes = boxes / train_detector.IM_SCALE

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

            skip_pixel_loss = False
            total_loss, losses =  calculate_model_losses(
                                                                args, skip_pixel_loss, model, imgs, imgs_pred)

            # total_iou += jaccard(boxes_pred, boxes)
            # total_boxes += boxes_pred.size(0)

            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)
            if num_samples >= args.num_val_samples:
                break

        samples = {}
        samples['gt_img'] = imgs

        # model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
        # samples['gt_box_gt_mask'] = model_out[0]

        # model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
        # samples['gt_box_pred_mask'] = model_out[0]
        #
        # model_out = model(objs, triples, obj_to_img)
        # samples['pred_box_pred_mask'] = model_out[0]

        model_out = model(obj_to_img, boxes, obj_fmap)
        samples['gt_box_gt_feat'] = model_out

        for k, images in samples.items():
            # samples[k] = imagenet_deprocess_batch(v, rescale=False)
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
            images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
            images = images - images_min
            images = images / (images_max - images_min)
            images = images.clamp(min=0, max=1)
            samples[k] = images

        mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
        # avg_iou = total_iou / total_boxes

        # masks_to_store = masks
        # if masks_to_store is not None:
        #   masks_to_store = masks_to_store.data.cpu().clone()

        # masks_pred_to_store = masks_pred
        # if masks_pred_to_store is not None:
        #   masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

    batch_data = {
        'objs': objs.detach().cpu().clone(),
        'boxes_gt': boxes.detach().cpu().clone(), 
        # 'masks_gt': masks_to_store,
        # 'triples': triples.detach().cpu().clone(),
        'obj_to_img': obj_to_img.detach().cpu().clone(),
        # 'triple_to_img': triple_to_img.detach().cpu().clone(),
        # 'boxes_pred': boxes_pred.detach().cpu().clone(),
        # 'masks_pred': masks_pred_to_store
    }
    out = [mean_losses, samples, batch_data] #, avg_iou]

    return tuple(out)


def calculate_model_losses(args, skip_pixel_loss, model, img, img_pred):
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_weight = args.l1_pixel_loss_weight * (1 - skip_pixel_loss)
    # if skip_pixel_loss:
    #   l1_pixel_weight = 0
    l1_pixel_loss = F.l1_loss(img_pred, img)
    total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                                                l1_pixel_weight)
    # loss_bbox = F.mse_loss(bbox_pred, bbox)
    # total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
    #                       args.bbox_pred_loss_weight)

    # if args.predicate_pred_loss_weight > 0:
    #   loss_predicate = F.cross_entropy(predicate_scores, predicates)
    #   total_loss = add_loss(total_loss, loss_predicate, losses, 'predicate_pred',
    #                         args.predicate_pred_loss_weight)
    #
    # if args.mask_loss_weight > 0 and masks is not None and masks_pred is not None:
    #   mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    #   total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
    #                         args.mask_loss_weight)
    return total_loss, losses


def main(args):
    print(args)
    check_args(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    detector_gather_device = args.num_gpus - 1
    sg2im_device = torch.device(args.num_gpus - 1)
    args.detector_gather_device = detector_gather_device
    args.sg2im_device = sg2im_device
    if not exists(args.output_dir):
            os.makedirs(args.output_dir)
    summary_writer = SummaryWriter(args.output_dir)

    # vocab, train_loader, val_loader = build_loaders(args)
    # self.ind_to_classes, self.ind_to_predicates
    vocab = {
        'object_idx_to_name': train_detector.train.ind_to_classes,
    }
    model, model_kwargs = build_model(args) #, vocab)print(type(batch.imgs), len(batch.imgs), type(batch.imgs[0]))

    # model.type(float_dtype)
    model = model.to(sg2im_device)
    # model = DataParallel(model, list(range(args.num_gpus)))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
    img_discriminator, d_img_kwargs = build_img_discriminator(args) #, vocab)
    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    if obj_discriminator is not None:
        # obj_discriminator.type(float_dtype)
        obj_discriminator = obj_discriminator.to(sg2im_device)
        # obj_discriminator = DataParallel(obj_discriminator, list(range(args.num_gpus)))
        obj_discriminator.train()
        print(obj_discriminator)
        optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(),
                                                                             lr=args.learning_rate)

    if img_discriminator is not None:
        # img_discriminator.type(float_dtype)
        img_discriminator = img_discriminator.to(sg2im_device)
        # img_discriminator = DataParallel(img_discriminator, list(range(args.num_gpus)))
        img_discriminator.train()
        print(img_discriminator)
        optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(),
                                                                             lr=args.learning_rate)

    restore_path = None
    if args.restore_from_checkpoint:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
    if restore_path is not None and os.path.isfile(restore_path):
        print('Restoring from checkpoint:')
        print(restore_path)
        checkpoint = torch.load(restore_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])

        if obj_discriminator is not None:
            obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
            optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

        if img_discriminator is not None:
            img_discriminator.load_state_dict(checkpoint['d_img_state'])
            optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

        t = checkpoint['counters']['t']
        if 0 <= args.eval_mode_after <= t:
            model.eval()
        else:
            model.train()
        epoch = checkpoint['counters']['epoch']
    else:
        t, epoch = 0, 0
        checkpoint = {
            # 'args': args.__dict__,
            'vocab': vocab,
            'model_kwargs': model_kwargs,
            'd_obj_kwargs': d_obj_kwargs,
            'd_img_kwargs': d_img_kwargs,
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_batch_data': [], 
            'train_samples': [],
            'train_iou': [],
            'val_batch_data': [], 
            'val_samples': [],
            'val_losses': defaultdict(list),
            'val_iou': [], 
            'norm_d': [], 
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None, 'optim_state': None,
            'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
            'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
            'best_t': [],
        }

    while True:
        if t >= args.num_iterations:
            break
        epoch += 1
        print('Starting epoch %d' % epoch)
        
        # for batch in train_loader:
        # for batch in train_detector.train_loader:
        for step, batch in enumerate(tqdm(train_detector.train_loader, desc='Training Epoch %d' % epoch,
                                                                                    total=len(train_detector.train_loader))):
            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            t += 1
            # batch = [tensor.cuda() for tensor in batch]
            # masks = None
            # if len(batch) == 6:
            #   imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
            # elif len(batch) == 7:
            #   imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
            # else:
            #   assert False
            # predicates = triples[:, 1]

            with timeit('forward', args.timing):
                # model_boxes = boxes
                # model_masks = masks
                # model_out = model(objs, triples, obj_to_img,
                #                   boxes_gt=model_boxes, masks_gt=model_masks)
                # imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
                imgs = F.interpolate(batch.imgs, size=args.image_size).to(sg2im_device)
                with torch.no_grad():
                    if args.num_gpus > 2:
                        result = train_detector.detector.__getitem__(batch, target_device=detector_gather_device)
                    else:
                        result = train_detector.detector[batch]
                objs = result.obj_preds
                boxes = result.rm_box_priors
                obj_to_img = result.im_inds
                obj_fmap = result.obj_fmap
                if args.num_gpus == 2:
                    objs = objs.to(sg2im_device)
                    boxes = boxes.to(sg2im_device)
                    obj_to_img = obj_to_img.to(sg2im_device)
                    obj_fmap = obj_fmap.to(sg2im_device)

                boxes /= train_detector.IM_SCALE
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

                # assert (cnt > 0).sum() == len(imgs), "some imgs have no detection"
                model_out = model(obj_to_img, boxes, obj_fmap)
                imgs_pred = model_out

            with timeit('loss', args.timing):
                # Skip the pixel loss if using GT boxes
                # skip_pixel_loss = (model_boxes is None)
                skip_pixel_loss = False
                total_loss, losses = calculate_model_losses(
                                                                args, skip_pixel_loss, model, imgs, imgs_pred)

            if obj_discriminator is not None:
                scores_fake, ac_loss = obj_discriminator(imgs_pred, objs, boxes, obj_to_img)
                total_loss = add_loss(total_loss, ac_loss, losses, 'ac_loss',
                                                            args.ac_loss_weight)
                weight = args.discriminator_loss_weight * args.d_obj_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                                            'g_gan_obj_loss', weight)

            if img_discriminator is not None:
                scores_fake = img_discriminator(imgs_pred)
                weight = args.discriminator_loss_weight * args.d_img_weight
                total_loss = add_loss(total_loss, gan_g_loss(scores_fake), losses,
                                                            'g_gan_img_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()
            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}
            
            if obj_discriminator is not None:
                d_obj_losses = LossManager()
                imgs_fake = imgs_pred.detach()
                scores_fake, ac_loss_fake = obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
                scores_real, ac_loss_real = obj_discriminator(imgs, objs, boxes, obj_to_img)

                d_obj_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
                d_obj_losses.add_loss(ac_loss_real, 'd_ac_loss_real')
                d_obj_losses.add_loss(ac_loss_fake, 'd_ac_loss_fake')

                optimizer_d_obj.zero_grad()
                d_obj_losses.total_loss.backward()
                optimizer_d_obj.step()

            if img_discriminator is not None:
                d_img_losses = LossManager()
                imgs_fake = imgs_pred.detach()
                scores_fake = img_discriminator(imgs_fake)
                scores_real = img_discriminator(imgs)

                d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
                d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
                
                optimizer_d_img.zero_grad()
                d_img_losses.total_loss.backward()
                optimizer_d_img.step()

            if t % args.print_every == 0:
                print('t = %d / %d' % (t, args.num_iterations))
                G_loss_list = []
                for name, val in losses.items():
                    # print(' G [%s]: %.4f' % (name, val))
                    G_loss_list.append('[%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                    summary_writer.add_scalar("G_%s" % name, val, t)
                print("G: %s" % ", ".join(G_loss_list))
                checkpoint['losses_ts'].append(t)

                if obj_discriminator is not None:
                    D_obj_loss_list = []
                    for name, val in d_obj_losses.items():
                        # print(' D_obj [%s]: %.4f' % (name, val))
                        D_obj_loss_list.append('[%s]: %.4f' % (name, val))
                        checkpoint['d_losses'][name].append(val)
                        summary_writer.add_scalar("D_obj_%s" % name, val, t)
                    print("D_obj: %s" % ", ".join(D_obj_loss_list))

                if img_discriminator is not None:
                    D_img_loss_list = []
                    for name, val in d_img_losses.items():
                        # print(' D_img [%s]: %.4f' % (name, val))
                        D_img_loss_list.append('[%s]: %.4f' % (name, val))
                        checkpoint['d_losses'][name].append(val)
                        summary_writer.add_scalar("D_img_%s" % name, val, t)
                    print("D_img: %s" % ", ".join(D_img_loss_list))
            
            if t % args.checkpoint_every == 0:
                print('checking on train')
                train_results = check_model(args, t, train_detector.train_loader, model)
                # t_losses, t_samples, t_batch_data, t_avg_iou = train_results
                t_losses, t_samples, t_batch_data = train_results

                checkpoint['train_batch_data'].append(t_batch_data)
                checkpoint['train_samples'].append(t_samples)
                checkpoint['checkpoint_ts'].append(t)
                # checkpoint['train_iou'].append(t_avg_iou)
                for name, images in t_samples.items():
                    # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                    # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                    summary_writer.add_image("train_%s" % name, images, t)

                print('checking on val')
                val_results = check_model(args, t, train_detector.val_loader, model)
                # val_losses, val_samples, val_batch_data, val_avg_iou = val_results
                val_losses, val_samples, val_batch_data = val_results
                checkpoint['val_samples'].append(val_samples)
                checkpoint['val_batch_data'].append(val_batch_data)
                # checkpoint['val_iou'].append(val_avg_iou)
                for name, images in val_samples.items():
                    # images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                    # images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                    summary_writer.add_image("val_%s" % name, images, t)

                # print('train iou: ', t_avg_iou)
                # print('val iou: ', val_avg_iou)

                for k, v in val_losses.items():
                    checkpoint['val_losses'][k].append(v)
                    summary_writer.add_scalar("val_%s" % k, v, t)
                checkpoint['model_state'] = model.state_dict()

                if obj_discriminator is not None:
                    checkpoint['d_obj_state'] = obj_discriminator.state_dict()
                    checkpoint['d_obj_optim_state'] = optimizer_d_obj.state_dict()

                if img_discriminator is not None:
                    checkpoint['d_img_state'] = img_discriminator.state_dict()
                    checkpoint['d_img_optim_state'] = optimizer_d_img.state_dict()

                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(args.output_dir,
                                                            '%s_with_model.pt' % args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

                # Save another checkpoint without any model or optim state
                checkpoint_path = os.path.join(args.output_dir,
                                                            '%s_no_model.pt' % args.checkpoint_name)
                key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                                                 'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                                                 'd_img_state', 'd_img_optim_state', 'd_img_best_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)


if __name__ == '__main__':
    args = train_detector.conf
    main(args)

