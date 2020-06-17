from lib.object_detector import ObjectDetector, gather_res
import torch
import torch.nn as nn
import torch.nn.parallel
from sg2im_model_bicyclegan import Sg2ImModel
from sg2im.layout import boxes_to_layout
from discriminators import PatchDiscriminator, AcCropDiscriminator
from encoder import ImageEncoder
import os
from collections import defaultdict
from lib.pytorch_misc import optimistic_restore
import torch.nn.functional as F
from config import BOX_SCALE
from sg2im.utils import timeit
from sg2im.losses import gradient_penalty
from sg2im.bilinear import crop_bbox_batch


def build_model(args):
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
            'args': args,
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
        model = Sg2ImModel(**kwargs).cuda()
    return model, kwargs


def build_img_encoder(args):
    if args.checkpoint_start_from is not None:
        checkpoint = torch.load(args.checkpoint_start_from)
        e_img_kwargs = checkpoint['e_img_kwargs']
        img_encoder = ImageEncoder(**e_img_kwargs)
        raw_state_dict = checkpoint['e_img_state']
        state_dict = {}
        for k, v in raw_state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            state_dict[k] = v
        img_encoder.load_state_dict(state_dict)
    else:
        e_img_kwargs = {
            'arch': args.d_img_arch,
            'normalization': args.d_normalization,
            'activation': args.d_activation,
            'padding': args.d_padding,
            'args': args
        }
        img_encoder = ImageEncoder(**e_img_kwargs).cuda()
    return img_encoder, e_img_kwargs


def build_obj_discriminator(args, vocab):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_obj_weight = args.d_obj_weight
    if (d_weight == 0 or d_obj_weight == 0) and args.ac_loss_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'vocab': vocab,
        'arch': args.d_obj_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'object_size': args.crop_size,
        'args': args
    }
    discriminator = AcCropDiscriminator(**d_kwargs).cuda()
    return discriminator, d_kwargs


def build_img_discriminator(args):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_img_weight = args.d_img_weight
    if d_weight == 0 or d_img_weight == 0:
        return discriminator, d_kwargs

    layout_dim = 0
    if args.condition_d_img:
        layout_dim = args.gconv_dim
        if args.condition_d_img_on_class_label_map:
            layout_dim = 3
    d_kwargs = {
        'arch': args.d_img_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'layout_dim': layout_dim,
        'args': args
    }
    discriminator = PatchDiscriminator(**d_kwargs).cuda()
    return discriminator, d_kwargs


def build_bg_discriminator(args):
    discriminator = None
    d_kwargs = {}
    d_weight = args.discriminator_loss_weight
    d_bg_weight = args.d_bg_weight
    if d_weight == 0 or d_bg_weight == 0:
        return discriminator, d_kwargs

    d_kwargs = {
        'arch': args.d_bg_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
        'layout_dim': 3 if args.condition_d_bg else 0,
        'args': args
    }
    discriminator = PatchDiscriminator(**d_kwargs).cuda()
    return discriminator, d_kwargs


class neural_motifs_sg2im_model(nn.Module):
    def __init__(self, args, ind_to_classes):
        super(neural_motifs_sg2im_model, self).__init__()
        self.args = args

        # define and initial generator, obj_encoder, image_discriminator, obj_discriminator
        # and corresponding optimizer
        vocab = {
                'object_idx_to_name': ind_to_classes,
        }
        self.model, model_kwargs = build_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        self.img_encoder, e_img_kwargs = build_img_encoder(args)
        self.optimizer_e_img = torch.optim.Adam(self.img_encoder.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        self.obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
        self.img_discriminator, d_img_kwargs = build_img_discriminator(args)
        self.bg_discriminator, d_bg_kwargs = build_bg_discriminator(args)

        if self.obj_discriminator is not None:
            self.obj_discriminator.train()
            self.optimizer_d_obj = torch.optim.Adam(self.obj_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if self.img_discriminator is not None:
            self.img_discriminator.train()
            self.optimizer_d_img = torch.optim.Adam(self.img_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if self.bg_discriminator is not None:
            self.bg_discriminator.train()
            self.optimizer_d_bg = torch.optim.Adam(self.bg_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        restore_path = None
        if args.restore_from_checkpoint:
            restore_path = '%s_with_model.pt' % args.checkpoint_name
            restore_path = os.path.join(args.output_dir, restore_path)
        if restore_path is not None and os.path.isfile(restore_path):
            print('Restoring from checkpoint:')
            print(restore_path)
            checkpoint = torch.load(restore_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optim_state'])

            self.img_encoder.load_state_dict(checkpoint['e_img_state'])
            self.optimizer_e_img.load_state_dict(checkpoint['e_img_optim_state'])

            if self.obj_discriminator is not None:
                self.obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
                self.optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

            if self.img_discriminator is not None:
                self.img_discriminator.load_state_dict(checkpoint['d_img_state'])
                self.optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

            if self.bg_discriminator is not None:
                self.bg_discriminator.load_state_dict(checkpoint['d_bg_state'])
                self.optimizer_d_bg.load_state_dict(checkpoint['d_bg_optim_state'])

            t = checkpoint['counters']['t']
            if 0 <= args.eval_mode_after <= t:
                self.model.eval()
            else:
                self.model.train()
            epoch = checkpoint['counters']['epoch']
        else:
            t, epoch = 0, 0
            checkpoint = {
                'vocab': vocab,
                'model_kwargs': model_kwargs,
                'd_obj_kwargs': d_obj_kwargs,
                'd_img_kwargs': d_img_kwargs,
                'd_bg_kwargs': d_bg_kwargs,
                'e_img_kwargs': e_img_kwargs,
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
                'd_bg_state': None, 'd_bg_best_state': None, 'd_bg_optim_state': None,
                'e_img_state': None, 'e_img_best_state': None, 'e_img_optim_state': None,
                'best_t': [],
            }

        self.t, self.epoch, self.checkpoint = t, epoch, checkpoint
        self.forward_G = True
        self.calc_G_D_loss = True
        self.forward_D = True

    def forward(self, imgs, img_offset, gt_boxes, gt_classes, gt_fmaps):
        obj_to_img = gt_classes[:, 0] - img_offset
        # print("obj_to_img.min(), obj_to_img.max(), len(imgs) {} {} {}".format(obj_to_img.min(), obj_to_img.max(), len(imgs)))
        assert obj_to_img.min() >= 0 and obj_to_img.max() < len(imgs), \
            "obj_to_img.min() >= 0 and obj_to_img.max() < len(imgs) is not satidfied: {} {} {}".format(obj_to_img.min(), obj_to_img.max(), len(imgs))
        boxes = gt_boxes
        obj_fmaps = gt_fmaps
        objs = gt_classes[:, 1]

        if self.args is not None:
            if self.args.exchange_feat_cls:
                print("exchange feature vectors and classes among bboxes")
                for img_ind in range(imgs.shape[0]):
                    ind = (obj_to_img == img_ind).nonzero()[:, 0]
                    # permute = ind[torch.randperm(len(ind))]
                    # obj_fmaps[ind] = obj_fmaps[permute]
                    permute_ind = ind[torch.randperm(len(ind))[:2]]
                    permute = permute_ind[[1, 0]]
                    obj_fmaps[permute_ind] = obj_fmaps[permute]
                    objs[permute_ind] = objs[permute]

            if self.args.change_bbox:
                print("change the position of bboxes")
                for img_ind in range(imgs.shape[0]):
                    ind = (obj_to_img == img_ind).nonzero()[:, 0]
                    ind = ind[torch.randperm(len(ind))[0]]
                    if boxes[ind][3] < 0.8:
                        print("move to bottom")
                        boxes[ind][1] += (1 - boxes[ind][3])
                        boxes[ind][3] = 1
                    elif boxes[ind][1] > 0.2:
                        print("move to top")
                        boxes[ind][3] -= boxes[ind][1]
                        boxes[ind][1] = 0
                    elif boxes[ind][0] > 0.2:
                        print("move to left")
                        boxes[ind][2] -= boxes[ind][0]
                        boxes[ind][0] = 0
                    elif boxes[ind][2] < 0.8:
                        print("move to right")
                        boxes[ind][0] += (1 - boxes[ind][2])
                        boxes[ind][2] = 1
                    else:
                        print("move to bottom right")
                        boxes[ind][1] += (1 - boxes[ind][3])
                        boxes[ind][3] = 1
                        boxes[ind][0] += (1 - boxes[ind][2])
                        boxes[ind][2] = 1

        # obj_to_img, boxes, obj_fmaps, mask_noise_indexes
        half_size = imgs.shape[0] // 2

        obj_index_encoded = []
        obj_index_random = []
        for ind in range(half_size):
            obj_index_encoded.append((obj_to_img == ind).nonzero()[:, 0])
        obj_index_encoded = torch.cat(obj_index_encoded)
        for ind in range(half_size, imgs.shape[0]):
            obj_index_random.append((obj_to_img == ind).nonzero()[:, 0])
        obj_index_random = torch.cat(obj_index_random)

        imgs_encoded = imgs[:half_size]
        obj_to_img_encoded = obj_to_img[obj_index_encoded]
        boxes_encoded = boxes[obj_index_encoded]
        obj_fmaps_encoded = obj_fmaps[obj_index_encoded]
        mask_noise_indexes_encoded = torch.randperm(half_size)[:int(self.args.noise_mask_ratio * half_size)].to(imgs.device)
        if len(mask_noise_indexes_encoded) == 0:
            mask_noise_indexes_encoded = None
        # crops_encoded = crop_bbox_batch(imgs_encoded, boxes_encoded, obj_to_img_encoded, self.args.crop_size)

        imgs_random = imgs[half_size:]
        obj_to_img_random = obj_to_img[obj_index_random] - half_size
        boxes_random = boxes[obj_index_random]
        obj_fmaps_random = obj_fmaps[obj_index_random]
        mask_noise_indexes_random = torch.randperm(imgs.shape[0] - half_size)\
            [:int(self.args.noise_mask_ratio * (imgs.shape[0] - half_size))].to(imgs.device)
        if len(mask_noise_indexes_random) == 0:
            mask_noise_indexes_random = None
        # crops_random = crop_bbox_batch(imgs_random, boxes_random, obj_to_img_random, self.args.crop_size)

        mask_noise_indexes = None
        if mask_noise_indexes_encoded is not None:
            mask_noise_indexes = mask_noise_indexes_encoded
        if mask_noise_indexes_random is not None:
            if mask_noise_indexes is not None:
                mask_noise_indexes = torch.cat([mask_noise_indexes, mask_noise_indexes_random + half_size])
            else:
                mask_noise_indexes = mask_noise_indexes_random + half_size

        if self.forward_G:
            with timeit('generator forward', self.args.timing):
                if self.training:
                    # mu_encoded, logvar_encoded = self.obj_encoder(crops_encoded)
                    mu_encoded, logvar_encoded = self.img_encoder(imgs_encoded)
                    std = logvar_encoded.mul(0.5).exp_()
                    eps = torch.randn((std.size(0), std.size(1)), dtype=std.dtype, device=std.device)
                    z_encoded = eps.mul(std).add_(mu_encoded)
                    z_random = torch.randn((imgs_random.shape[0], self.args.layout_noise_dim),
                                           dtype=imgs_random.dtype, device=imgs_random.device)

                    imgs_pred_encoded, layout_encoded = self.model(obj_to_img_encoded, boxes_encoded, obj_fmaps_encoded,
                                                                   mask_noise_indexes=mask_noise_indexes_encoded,
                                                                   object_noise=z_encoded)
                    imgs_pred_random, layout_random = self.model(obj_to_img_random, boxes_random, obj_fmaps_random,
                                                                 mask_noise_indexes=mask_noise_indexes_random,
                                                                 object_noise=z_random)

                    mu_rec, logvar_rec = self.obj_encoder(imgs_pred_random)
                    z_random_rec = mu_rec

                    imgs_pred = torch.cat([imgs_pred_encoded, imgs_pred_random], dim=0)

                    layout = torch.cat([layout_encoded, layout_random], dim=0).detach()
                else:
                    z_random = torch.randn((imgs.shape[0], self.args.layout_noise_dim),
                                           dtype=imgs.dtype, device=imgs.device)
                    imgs_pred, layout = self.model(obj_to_img, boxes, obj_fmaps,
                                                    mask_noise_indexes=mask_noise_indexes,
                                                    object_noise=z_random)
                    layout = layout.detach()
                    imgs_encoded = None
                    imgs_pred_encoded = None
                    z_random = None
                    z_random_rec = None
                    mu_encoded = None
                    logvar_encoded = None

        H, W = self.args.image_size
        bg_layout = boxes_to_layout(torch.ones(boxes.shape[0], 3).to(imgs.device), boxes, obj_to_img, H, W)
        bg_layout = (bg_layout <= 0).type(imgs.dtype)

        if self.args.condition_d_img_on_class_label_map:
            layout = boxes_to_layout((objs+1).view(-1, 1).repeat(1, 3).type(imgs.dtype), boxes, obj_to_img, H, W)

        g_scores_fake_crop, g_obj_scores_fake_crop, g_rec_feature_fake_crop = None, None, None
        g_scores_fake_img = None
        g_scores_fake_bg = None
        if self.calc_G_D_loss:
            # forward discriminators to train generator
            if self.obj_discriminator is not None:
                with timeit('d_obj forward for g', self.args.timing):
                    g_scores_fake_crop, g_obj_scores_fake_crop, _, g_rec_feature_fake_crop = \
                        self.obj_discriminator(imgs_pred, objs, boxes, obj_to_img)

            if self.img_discriminator is not None:
                with timeit('d_img forward for g', self.args.timing):
                    if self.args.condition_d_img:
                        g_scores_fake_img = self.img_discriminator(imgs_pred, layout)
                    else:
                        g_scores_fake_img = self.img_discriminator(imgs_pred)

            if self.bg_discriminator is not None:
                with timeit('d_bg forward for g', self.args.timing):
                    if self.args.condition_d_bg:
                        g_scores_fake_bg = self.bg_discriminator(imgs_pred, bg_layout)
                    else:
                        g_scores_fake_bg = self.bg_discriminator(imgs_pred * bg_layout)

        d_scores_fake_crop, d_obj_scores_fake_crop, fake_crops, d_rec_feature_fake_crop = None, None, None, None
        d_scores_real_crop, d_obj_scores_real_crop, real_crops, d_rec_feature_real_crop = None, None, None, None
        d_obj_gp = None
        d_scores_fake_img = None
        d_scores_real_img = None
        d_img_gp = None
        d_scores_fake_bg = None
        d_scores_real_bg = None
        d_bg_gp = None
        if self.forward_D:
            # forward discriminators to train discriminators
            if self.obj_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_obj forward for d', self.args.timing):
                    d_scores_fake_crop, d_obj_scores_fake_crop, fake_crops, d_rec_feature_fake_crop = \
                        self.obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
                    d_scores_real_crop, d_obj_scores_real_crop, real_crops, d_rec_feature_real_crop = \
                        self.obj_discriminator(imgs, objs, boxes, obj_to_img)
                    if self.args.gan_loss_type == "wgan-gp" and self.training:
                        d_obj_gp = gradient_penalty(real_crops.detach(), fake_crops.detach(), self.obj_discriminator.discriminator)

            if self.img_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_img forward for d', self.args.timing):
                    if self.args.condition_d_img:
                        d_scores_fake_img = self.img_discriminator(imgs_fake, layout)
                        d_scores_real_img = self.img_discriminator(imgs, layout)
                    else:
                        d_scores_fake_img = self.img_discriminator(imgs_fake)
                        d_scores_real_img = self.img_discriminator(imgs)

                    if self.args.gan_loss_type == "wgan-gp" and self.training:
                        if self.args.condition_d_img:
                            d_img_gp = gradient_penalty(torch.cat([imgs, layout], dim=1), torch.cat([imgs_fake, layout], dim=1), self.img_discriminator)
                        else:
                            d_img_gp = gradient_penalty(imgs, imgs_fake, self.img_discriminator)

            if self.bg_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_bg forward for d', self.args.timing):
                    if self.args.condition_d_bg:
                        d_scores_fake_bg = self.bg_discriminator(imgs_fake, bg_layout)
                        d_scores_real_bg = self.bg_discriminator(imgs, bg_layout)
                    else:
                        d_scores_fake_bg = self.bg_discriminator(imgs_fake * bg_layout)
                        d_scores_real_bg = self.bg_discriminator(imgs * bg_layout)

                    if self.args.gan_loss_type == "wgan-gp" and self.training:
                        if self.args.condition_d_bg:
                            d_bg_gp = gradient_penalty(torch.cat([imgs, bg_layout], dim=1), torch.cat([imgs_fake, bg_layout], dim=1), self.bg_discriminator)
                        else:
                            d_bg_gp = gradient_penalty(imgs * bg_layout, imgs_fake * bg_layout, self.bg_discriminator)
        return Result(
            imgs=imgs,
            imgs_pred=imgs_pred,
            objs=objs,
            obj_fmaps=obj_fmaps,
            boxes=boxes,
            obj_to_img=obj_to_img + img_offset,
            g_scores_fake_crop=g_scores_fake_crop,
            g_obj_scores_fake_crop=g_obj_scores_fake_crop,
            g_scores_fake_img=g_scores_fake_img,
            d_scores_fake_crop=d_scores_fake_crop,
            d_obj_scores_fake_crop=d_obj_scores_fake_crop,
            d_scores_real_crop=d_scores_real_crop,
            d_obj_scores_real_crop=d_obj_scores_real_crop,
            d_scores_fake_img=d_scores_fake_img,
            d_scores_real_img=d_scores_real_img,
            d_obj_gp=d_obj_gp,
            d_img_gp=d_img_gp,
            fake_crops=fake_crops,
            real_crops=real_crops,
            mask_noise_indexes=(mask_noise_indexes + img_offset) if mask_noise_indexes is not None else None,
            g_rec_feature_fake_crop=g_rec_feature_fake_crop,
            d_rec_feature_fake_crop=d_rec_feature_fake_crop,
            d_rec_feature_real_crop=d_rec_feature_real_crop,
            g_scores_fake_bg=g_scores_fake_bg,
            d_scores_fake_bg=d_scores_fake_bg,
            d_scores_real_bg=d_scores_real_bg,
            d_bg_gp=d_bg_gp,
            bg_layout=bg_layout,
            imgs_encoded=imgs_encoded,
            imgs_pred_encoded=imgs_pred_encoded,
            z_random=z_random,
            z_random_rec=z_random_rec,
            mu_encoded=mu_encoded,
            logvar_encoded=logvar_encoded
        )
        # return imgs, imgs_pred, objs, g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, d_scores_fake_crop, \
        #        d_obj_scores_fake_crop, d_scores_real_crop, d_obj_scores_real_crop, d_scores_fake_img, d_scores_real_img

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        if self.forward_G:
            batch.scatter()
        if self.args.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.args.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.args.num_gpus)])

        # if self.training:
        #     return gather_res(outputs, 0, dim=0)
        # return outputs
        return gather_res(outputs, 0, dim=0)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


class Result(object):
    def __init__(self, imgs=None,
            imgs_pred=None,
            objs=None,
            obj_fmaps=None,
            boxes=None,
            obj_to_img=None,
            g_scores_fake_crop=None,
            g_obj_scores_fake_crop=None,
            g_scores_fake_img=None,
            d_scores_fake_crop=None,
            d_obj_scores_fake_crop=None,
            d_scores_real_crop=None,
            d_obj_scores_real_crop=None,
            d_scores_fake_img=None,
            d_scores_real_img=None,
            d_obj_gp=None,
            d_img_gp=None,
            fake_crops=None,
            real_crops=None,
            mask_noise_indexes=None,
            g_rec_feature_fake_crop=None,
            d_rec_feature_fake_crop=None,
            d_rec_feature_real_crop=None,
            g_scores_fake_bg=None,
            d_scores_fake_bg=None,
            d_scores_real_bg=None,
            d_bg_gp=None,
            bg_layout=None,
            imgs_encoded=None,
            imgs_pred_encoded=None,
            z_random=None,
            z_random_rec=None,
            mu_encoded=None,
            logvar_encoded=None,
            ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([v is None for k, v in self.__dict__.items() if k != 'self'])
