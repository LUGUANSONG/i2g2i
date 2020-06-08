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
from sg2im.losses import get_gan_losses, VGGLoss
from sg2im.utils import timeit, bool_flag, LossManager

# neural motifs
# from dataloaders.visual_genome import VGDataLoader, VG
# from dataloaders.mscoco import CocoDetection, CocoDataLoader
from torchvision import transforms
from bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
# from config import ModelConfig
from config_args import config_args
from copy import deepcopy

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
    model.eval()
    num_samples = 0
    all_losses = defaultdict(list)
    model.forward_G = True
    model.calc_G_D_loss = False
    model.forward_D = True
    with torch.no_grad():
        for batch in loader:
            _batch = deepcopy(batch)
            result = model[_batch]
            # imgs, imgs_pred, objs, g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, \
            # d_scores_fake_crop, d_obj_scores_fake_crop, d_scores_real_crop, d_obj_scores_real_crop, \
            # d_scores_fake_img, d_scores_real_img = result.imgs, result.imgs_pred, result.objs, \
            # result.g_scores_fake_crop, result.g_obj_scores_fake_crop, result.g_scores_fake_img, \
            # result.d_scores_fake_crop, result.d_obj_scores_fake_crop, result.d_scores_real_crop, \
            # result.d_obj_scores_real_crop, result.d_scores_fake_img, result.d_scores_real_img
            imgs, imgs_pred = result.imgs, result.imgs_pred
            mask_noise_indexes = result.mask_noise_indexes

            total_loss, losses = calculate_model_losses(
                args, imgs, imgs_pred, mask_noise_indexes)
            for loss_name, loss_val in losses.items():
                all_losses[loss_name].append(loss_val)
            num_samples += imgs.size(0)
            if num_samples >= args.num_val_samples:
                break

        same_input_different_noise = []
        for i in range(args.num_diff_noise):
            _batch = deepcopy(batch)
            result = model[_batch]
            same_input_different_noise.append(result.imgs_pred.detach().cpu())
        different_same_input = [torch.cat([batch[i:i+1] for batch in same_input_different_noise], dim=3) for i in range(len(same_input_different_noise[0]))]
        different_same_input = torch.cat(different_same_input, dim=2)

        samples = {}
        samples['gt_img'] = imgs
        samples['pred_img'] = imgs_pred
        samples['diff_noise_img'] = different_same_input
        samples['bg_layout'] = result.bg_layout
        if model.obj_discriminator is not None:
            real_crops, fake_crops = result.real_crops, result.fake_crops
            samples['real_crops'] = real_crops
            samples['fake_crops'] = fake_crops

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


def calculate_model_losses(args, img, img_pred, mask_noise_indexes=None):
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_weight = args.l1_pixel_loss_weight
    if mask_noise_indexes is not None:
        l1_pixel_loss = F.l1_loss(img_pred[mask_noise_indexes], img[mask_noise_indexes])
    else:
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
    train, val, _ = VG.splits(transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ]))
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   num_gpus=args.num_gpus)
    print(train.ind_to_classes)

    all_in_one_model = neural_motifs_sg2im_model(args, train.ind_to_classes)
    print(all_in_one_model)
    # Freeze the detector
    # for n, param in all_in_one_model.detector.named_parameters():
    #     param.requires_grad = False
    all_in_one_model.cuda()
    gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)
    criterionVGG = VGGLoss() if args.perceptual_loss_weight > 0 else None

    t, epoch, checkpoint = all_in_one_model.t, all_in_one_model.epoch, all_in_one_model.checkpoint

    def D_step(result):
        imgs, imgs_pred, objs, \
        d_scores_fake_crop, d_obj_scores_fake_crop, d_scores_real_crop, \
        d_obj_scores_real_crop, d_scores_fake_img, d_scores_real_img, \
        d_obj_gp, d_img_gp \
        = result.imgs, result.imgs_pred, result.objs, \
          result.d_scores_fake_crop, result.d_obj_scores_fake_crop, result.d_scores_real_crop, \
          result.d_obj_scores_real_crop, result.d_scores_fake_img, result.d_scores_real_img, \
          result.d_obj_gp, result.d_img_gp
        d_rec_feature_fake_crop, d_rec_feature_real_crop = result.d_rec_feature_fake_crop, result.d_rec_feature_real_crop
        obj_fmaps=result.obj_fmaps
        d_scores_fake_bg, d_scores_real_bg, d_bg_gp = result.d_scores_fake_bg, result.d_scores_real_bg, result.d_bg_gp

        d_obj_losses, d_img_losses, d_bg_losses = None, None, None
        if all_in_one_model.obj_discriminator is not None:
            with timeit('d_obj loss', args.timing):
                d_obj_losses = LossManager()
                if args.d_obj_weight > 0:
                    d_obj_gan_loss = gan_d_loss(d_scores_real_crop, d_scores_fake_crop)
                    d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
                    if args.gan_loss_type == 'wgan-gp':
                        d_obj_losses.add_loss(d_obj_gp.mean(), 'd_obj_gp', args.d_obj_gp_weight)
                if args.ac_loss_weight > 0:
                    d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_real_crop, objs), 'd_ac_loss_real')
                    d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_fake_crop, objs), 'd_ac_loss_fake')
                if args.d_obj_rec_feat_weight > 0:
                    d_obj_losses.add_loss(F.l1_loss(d_rec_feature_fake_crop, obj_fmaps), 'd_obj_fea_rec_loss_fake')
                    d_obj_losses.add_loss(F.l1_loss(d_rec_feature_real_crop, obj_fmaps), 'd_obj_fea_rec_loss_real')

            with timeit('d_obj backward', args.timing):
                all_in_one_model.optimizer_d_obj.zero_grad()
                d_obj_losses.total_loss.backward()
                all_in_one_model.optimizer_d_obj.step()

        if all_in_one_model.img_discriminator is not None:
            with timeit('d_img loss', args.timing):
                d_img_losses = LossManager()
                d_img_gan_loss = gan_d_loss(d_scores_real_img, d_scores_fake_img)
                d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
                if args.gan_loss_type == 'wgan-gp':
                    d_img_losses.add_loss(d_img_gp.mean(), 'd_img_gp', args.d_img_gp_weight)

            with timeit('d_img backward', args.timing):
                all_in_one_model.optimizer_d_img.zero_grad()
                d_img_losses.total_loss.backward()
                all_in_one_model.optimizer_d_img.step()

        if all_in_one_model.bg_discriminator is not None:
            with timeit('d_bg loss', args.timing):
                d_bg_losses = LossManager()
                d_bg_gan_loss = gan_d_loss(d_scores_real_bg, d_scores_fake_bg)
                d_bg_losses.add_loss(d_bg_gan_loss, 'd_bg_gan_loss')
                if args.gan_loss_type == 'wgan-gp':
                    d_bg_losses.add_loss(d_bg_gp.mean(), 'd_bg_gp', args.d_bg_gp_weight)

            with timeit('d_bg backward', args.timing):
                all_in_one_model.optimizer_d_bg.zero_grad()
                d_bg_losses.total_loss.backward()
                all_in_one_model.optimizer_d_bg.step()

        return d_obj_losses, d_img_losses, d_bg_losses

    def G_step(result):
        imgs, imgs_pred, objs, \
        g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, \
        = result.imgs, result.imgs_pred, result.objs, \
          result.g_scores_fake_crop, result.g_obj_scores_fake_crop, result.g_scores_fake_img
        mask_noise_indexes = result.mask_noise_indexes
        g_rec_feature_fake_crop = result.g_rec_feature_fake_crop
        obj_fmaps = result.obj_fmaps
        g_scores_fake_bg = result.g_scores_fake_bg
        
        with timeit('loss', args.timing):
            total_loss, losses = calculate_model_losses(
                args, imgs, imgs_pred, mask_noise_indexes)

            if criterionVGG is not None:
                if mask_noise_indexes is not None and args.perceptual_not_on_noise:
                    perceptual_loss = criterionVGG(imgs_pred[mask_noise_indexes], imgs[mask_noise_indexes])
                else:
                    perceptual_loss = criterionVGG(imgs_pred, imgs)
                total_loss = add_loss(total_loss, perceptual_loss, losses, 'perceptual_loss',
                                      args.perceptual_loss_weight)

            if all_in_one_model.obj_discriminator is not None:
                total_loss = add_loss(total_loss, F.cross_entropy(g_obj_scores_fake_crop, objs), losses, 'ac_loss',
                                      args.ac_loss_weight)
                weight = args.discriminator_loss_weight * args.d_obj_weight
                total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_crop), losses,
                                      'g_gan_obj_loss', weight)
                if args.d_obj_rec_feat_weight > 0:
                    total_loss = add_loss(total_loss, F.l1_loss(g_rec_feature_fake_crop, obj_fmaps), losses,
                                          'g_obj_fea_rec_loss', args.d_obj_rec_feat_weight)

            if all_in_one_model.img_discriminator is not None:
                weight = args.discriminator_loss_weight * args.d_img_weight
                total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_img), losses,
                                      'g_gan_img_loss', weight)

            if all_in_one_model.bg_discriminator is not None:
                weight = args.discriminator_loss_weight * args.d_bg_weight
                total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_bg), losses,
                                      'g_gan_bg_loss', weight)

        losses['total_loss'] = total_loss.item()

        if math.isfinite(losses['total_loss']):
            with timeit('backward', args.timing):
                all_in_one_model.optimizer.zero_grad()
                total_loss.backward()
                all_in_one_model.optimizer.step()

        return losses

    while True:
        if t >= args.num_iterations * (args.n_critic + args.n_gen):
            break
        epoch += 1
        print('Starting epoch %d' % epoch)

        for step, batch in enumerate(tqdm(train_loader, desc='Training Epoch %d' % epoch, total=len(train_loader))):
            # if t == args.eval_mode_after:
            #     print('switching to eval mode')
            #     all_in_one_model.model.eval()
            #     all_in_one_model.optimizer = optim.Adam(all_in_one_model.parameters(), lr=args.learning_rate)
            all_in_one_model.train()
            modes = ['l1', 'noise_std', 'd_obj', 'd_img', 'ac_loss']
            attrs = ['l1_pixel_loss_weight', 'noise_std', 'd_obj_weight', 'd_img_weight', 'ac_loss_weight']
            for mode, attr in zip(modes, attrs):
                old_value = getattr(args, attr)
                if getattr(args, "%s_mode" % mode) == "change" and t in getattr(args, "%s_change_iters" % mode):
                    step_index = getattr(args, "%s_change_iters" % mode).index(t)
                    new_value = getattr(args, "%s_change_vals" % mode)[step_index]
                    setattr(args, attr, new_value)
                    print("Change %s from %.10f to %.10f at iteration %d" % (attr, old_value, getattr(args, attr), t))
                elif getattr(args, "%s_mode" % mode) == "change_linear":
                    start_step = getattr(args, "%s_change_iters" % mode)[0]
                    end_step = getattr(args, "%s_change_iters" % mode)[1]
                    if start_step <= t <= end_step:
                        start_val = getattr(args, "%s_change_vals" % mode)[0]
                        end_val = getattr(args, "%s_change_vals" % mode)[1]
                        new_value = start_val + (end_val - start_val) * (t - start_step) / (end_step - start_step)
                        setattr(args, attr, new_value)
                        print("Change %s from %.10f to %.10f at iteration %d" % (attr, old_value, getattr(args, attr), t))
                    elif t > end_step:
                        end_val = getattr(args, "%s_change_vals" % mode)[1]
                        if old_value != end_val:
                            new_value = end_val
                            setattr(args, attr, new_value)
                            print("probably resume training from previous checkpoint")
                            print("Change %s from %.10f to %.10f at iteration %d" % (attr, old_value, getattr(args, attr), t))
            t += 1
            if args.gan_loss_type in ["wgan", "wgan-gp"] or args.n_critic != 0:
                # train discriminator (critic) for n_critic iterations
                if t % (args.n_critic + args.n_gen) in list(range(1, args.n_critic+1)):
                    all_in_one_model.forward_G = True
                    all_in_one_model.calc_G_D_loss = False
                    all_in_one_model.forward_D = True
                    all_in_one_model.set_requires_grad(
                        [all_in_one_model.obj_discriminator, all_in_one_model.img_discriminator],
                        True)
                    with timeit('forward', args.timing):
                        result = all_in_one_model[batch]
                    d_obj_losses, d_img_losses, d_bg_losses = D_step(result)

                # train generator for 1 iteration after n_critic iterations
                if t % (args.n_critic + args.n_gen) in (list(range(args.n_critic+1, args.n_critic + args.n_gen)) + [0]):
                    all_in_one_model.forward_G = True
                    all_in_one_model.calc_G_D_loss = True
                    all_in_one_model.forward_D = False
                    all_in_one_model.set_requires_grad(
                        [all_in_one_model.obj_discriminator, all_in_one_model.img_discriminator],
                        False)
                    result = all_in_one_model[batch]

                    losses = G_step(result)
                    if not math.isfinite(losses['total_loss']):
                        print('WARNING: Got loss = NaN, not backpropping')
                        continue
            else: # vanilla gan or lsgan
                all_in_one_model.forward_G = True
                all_in_one_model.calc_G_D_loss = True
                all_in_one_model.forward_D = True
                with timeit('forward', args.timing):
                    result = all_in_one_model[batch]
                losses = G_step(result)
                if not math.isfinite(losses['total_loss']):
                    print('WARNING: Got loss = NaN, not backpropping')
                    continue
                d_obj_losses, d_img_losses, d_bg_losses = D_step(result)

            if t % (args.print_every * (args.n_critic + args.n_gen)) == 0:
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

                if all_in_one_model.bg_discriminator is not None:
                    D_bg_loss_list = []
                    for name, val in d_bg_losses.items():
                        D_bg_loss_list.append('[%s]: %.4f' % (name, val))
                        checkpoint['d_losses'][name].append(val)
                        summary_writer.add_scalar("D_bg_%s" % name, val, t)
                    print("D_bg: %s" % ", ".join(D_bg_loss_list))

            if t % (args.checkpoint_every * (args.n_critic + args.n_gen)) == 0:
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

                if all_in_one_model.bg_discriminator is not None:
                    checkpoint['d_bg_state'] = all_in_one_model.bg_discriminator.state_dict()
                    checkpoint['d_bg_optim_state'] = all_in_one_model.optimizer_d_bg.state_dict()

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
                                 'd_img_state', 'd_img_optim_state', 'd_img_best_state',
                                 'd_bg_state', 'd_bg_optim_state', 'd_bg_best_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)


if __name__ == '__main__':
    # args = ModelConfig()
    args = config_args
    main(args)

