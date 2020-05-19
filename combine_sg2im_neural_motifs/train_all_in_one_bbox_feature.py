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

import os
from os.path import exists, join
import math
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# sg2im
from sg2im.losses import get_gan_losses
from sg2im.utils import timeit, bool_flag, LossManager

# neural motifs
# from dataloaders.visual_genome import VGDataLoader, VG
# from dataloaders.mscoco import CocoDetection, CocoDataLoader
from torchvision import transforms
from bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
# from config import ModelConfig
from config import config_args

# combine
from model_bbox_feature import neural_motifs_sg2im_model

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


def check_model(args, loader, model):
    num_samples = 0
    all_losses = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            result = model[batch]
            imgs, imgs_pred, objs, g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, \
            d_scores_fake_crop, d_obj_scores_fake_crop, d_scores_real_crop, d_obj_scores_real_crop, \
            d_scores_fake_img, d_scores_real_img = result.imgs, result.imgs_pred, result.objs, \
            result.g_scores_fake_crop, result.g_obj_scores_fake_crop, result.g_scores_fake_img, \
            result.d_scores_fake_crop, result.d_obj_scores_fake_crop, result.d_scores_real_crop, \
            result.d_obj_scores_real_crop, result.d_scores_fake_img, result.d_scores_real_img

            total_loss, losses = calculate_model_losses(
                args, imgs, imgs_pred)

            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)
            if num_samples >= args.num_val_samples:
                break

        samples = {}
        samples['gt_img'] = imgs
        samples['pred_img'] = imgs_pred

        for k, images in samples.items():
            images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
            images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
            images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
            images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
            images = images - images_min
            images = images / (images_max - images_min)
            images = images.clamp(min=0, max=1)
            samples[k] = images

        mean_losses = {k: np.mean(v) for k, v in all_losses.items()}

    out = [mean_losses, samples]

    return tuple(out)


def calculate_model_losses(args, img, img_pred):
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_weight = args.l1_pixel_loss_weight
    l1_pixel_loss = F.l1_loss(img_pred, img)
    # print("check l1_pixel_weight here, it is %.10f" % l1_pixel_weight)
    total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss',
                          l1_pixel_weight)
    return total_loss, losses


def main(args):
    print(args)
    check_args(args)
    if not exists(args.output_dir):
        os.makedirs(args.output_dir)
    summary_writer = SummaryWriter(args.output_dir)

    # if args.coco:
    #     train, val = CocoDetection.splits()
    #     val.ids = val.ids[:args.val_size]
    #     train.ids = train.ids
    #     train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=args.batch_size,
    #                                                      num_workers=args.num_workers,
    #                                                      num_gpus=args.num_gpus)
    # else:
    train, val = VG.splits(transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ]))
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   num_gpus=args.num_gpus)
    print(train.ind_to_classes)

    all_in_one_model = neural_motifs_sg2im_model(args, train.ind_to_classes)
    # Freeze the detector
    # for n, param in all_in_one_model.detector.named_parameters():
    #     param.requires_grad = False
    all_in_one_model.cuda()
    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

    t, epoch, checkpoint = all_in_one_model.t, all_in_one_model.epoch, all_in_one_model.checkpoint
    while True:
        if t >= args.num_iterations:
            break
        epoch += 1
        print('Starting epoch %d' % epoch)

        for step, batch in enumerate(tqdm(train_loader, desc='Training Epoch %d' % epoch, total=len(train_loader))):
            if t == args.eval_mode_after:
                print('switching to eval mode')
                all_in_one_model.model.eval()
                all_in_one_model.optimizer = optim.Adam(all_in_one_model.parameters(), lr=args.learning_rate)
            if args.l1_mode == "change":
                if t in args.l1_change_iters:
                    old_l1_weight = args.l1_pixel_loss_weight
                    args.l1_pixel_loss_weight = args.l1_change_vals[args.l1_change_iters.index(t)]
                    print("Change l1_pixel_loss_weight from %10.f to %.10f at iteration %d" % (
                    old_l1_weight, args.l1_pixel_loss_weight, t))
            if args.noise_std_mode == "change":
                if t in args.noise_std_change_iters:
                    old_noise_std = args.noise_std
                    args.noise_std = args.noise_std_change_vals[args.noise_std_change_iters.index(t)]
                    print("Change noise_std from %.10f to %.10f at iteration %d" % (old_noise_std, args.noise_std, t))
            t += 1

            with timeit('forward', args.timing):
                result = all_in_one_model[batch]
                imgs, imgs_pred, objs, g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, \
                d_scores_fake_crop, d_obj_scores_fake_crop, d_scores_real_crop, d_obj_scores_real_crop, \
                d_scores_fake_img, d_scores_real_img = result.imgs, result.imgs_pred, result.objs, \
                result.g_scores_fake_crop, result.g_obj_scores_fake_crop, result.g_scores_fake_img, \
                result.d_scores_fake_crop, result.d_obj_scores_fake_crop, result.d_scores_real_crop, \
                result.d_obj_scores_real_crop, result.d_scores_fake_img, result.d_scores_real_img

            with timeit('loss', args.timing):
                total_loss, losses = calculate_model_losses(
                    args, imgs, imgs_pred)

                if all_in_one_model.obj_discriminator is not None:
                    total_loss = add_loss(total_loss, F.cross_entropy(g_obj_scores_fake_crop, objs), losses, 'ac_loss',
                                          args.ac_loss_weight)
                    weight = args.discriminator_loss_weight * args.d_obj_weight
                    total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_crop), losses,
                                          'g_gan_obj_loss', weight)

                if all_in_one_model.img_discriminator is not None:
                    weight = args.discriminator_loss_weight * args.d_img_weight
                    total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_img), losses,
                                          'g_gan_img_loss', weight)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            with timeit('backward', args.timing):
                all_in_one_model.optimizer.zero_grad()
                total_loss.backward()
                all_in_one_model.optimizer.step()


            if all_in_one_model.obj_discriminator is not None:
                with timeit('d_obj loss', args.timing):
                    d_obj_losses = LossManager()
                    d_obj_gan_loss = gan_d_loss(d_obj_scores_real_crop, d_obj_scores_fake_crop)
                    d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
                    d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_real_crop, objs), 'd_ac_loss_real')
                    d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_fake_crop, objs), 'd_ac_loss_fake')

                with timeit('d_obj backward', args.timing):
                    all_in_one_model.optimizer_d_obj.zero_grad()
                    d_obj_losses.total_loss.backward()
                    all_in_one_model.optimizer_d_obj.step()

            if all_in_one_model.img_discriminator is not None:
                with timeit('d_img loss', args.timing):
                    d_img_losses = LossManager()
                    d_img_gan_loss = gan_d_loss(d_scores_real_img, d_scores_fake_img)
                    d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')

                with timeit('d_img backward', args.timing):
                    all_in_one_model.optimizer_d_img.zero_grad()
                    d_img_losses.total_loss.backward()
                    all_in_one_model.optimizer_d_img.step()

            if t % args.print_every == 0:
                print('t = %d / %d' % (t, args.num_iterations))
                G_loss_list = []
                for name, val in losses.items():
                    G_loss_list.append('[%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                    summary_writer.add_scalar("G_%s" % name, val, t)
                print("G: %s" % ", ".join(G_loss_list))
                checkpoint['losses_ts'].append(t)

                if all_in_one_model.obj_discriminator is not None:
                    D_obj_loss_list = []
                    for name, val in d_obj_losses.items():
                        D_obj_loss_list.append('[%s]: %.4f' % (name, val))
                        checkpoint['d_losses'][name].append(val)
                        summary_writer.add_scalar("D_obj_%s" % name, val, t)
                    print("D_obj: %s" % ", ".join(D_obj_loss_list))

                if all_in_one_model.img_discriminator is not None:
                    D_img_loss_list = []
                    for name, val in d_img_losses.items():
                        D_img_loss_list.append('[%s]: %.4f' % (name, val))
                        checkpoint['d_losses'][name].append(val)
                        summary_writer.add_scalar("D_img_%s" % name, val, t)
                    print("D_img: %s" % ", ".join(D_img_loss_list))

            if t % args.checkpoint_every == 0:
                print('checking on train')
                train_results = check_model(args, train_loader, all_in_one_model)
                t_losses, t_samples = train_results

                checkpoint['train_samples'].append(t_samples)
                checkpoint['checkpoint_ts'].append(t)
                for name, images in t_samples.items():
                    summary_writer.add_image("train_%s" % name, images, t)

                print('checking on val')
                val_results = check_model(args, val_loader, all_in_one_model)
                val_losses, val_samples = val_results
                checkpoint['val_samples'].append(val_samples)
                for name, images in val_samples.items():
                    summary_writer.add_image("val_%s" % name, images, t)

                for k, v in val_losses.items():
                    checkpoint['val_losses'][k].append(v)
                    summary_writer.add_scalar("val_%s" % k, v, t)
                checkpoint['model_state'] = all_in_one_model.model.state_dict()

                if all_in_one_model.obj_discriminator is not None:
                    checkpoint['d_obj_state'] = all_in_one_model.obj_discriminator.state_dict()
                    checkpoint['d_obj_optim_state'] = all_in_one_model.optimizer_d_obj.state_dict()

                if all_in_one_model.img_discriminator is not None:
                    checkpoint['d_img_state'] = all_in_one_model.img_discriminator.state_dict()
                    checkpoint['d_img_optim_state'] = all_in_one_model.optimizer_d_img.state_dict()

                checkpoint['optim_state'] = all_in_one_model.optimizer.state_dict()
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
    # args = ModelConfig()
    args = config_args
    main(args)

