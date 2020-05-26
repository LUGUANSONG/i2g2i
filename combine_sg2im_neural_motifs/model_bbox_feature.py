from lib.object_detector import ObjectDetector, gather_res
import torch
import torch.nn as nn
import torch.nn.parallel
from sg2im_model import Sg2ImModel
from discriminators import PatchDiscriminator, AcCropDiscriminator
import os
from collections import defaultdict
from lib.pytorch_misc import optimistic_restore
import torch.nn.functional as F
from config import BOX_SCALE
from sg2im.utils import timeit
from sg2im.losses import gradient_penalty


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

    d_kwargs = {
        'arch': args.d_img_arch,
        'normalization': args.d_normalization,
        'activation': args.d_activation,
        'padding': args.d_padding,
    }
    discriminator = PatchDiscriminator(**d_kwargs).cuda()
    return discriminator, d_kwargs


class neural_motifs_sg2im_model(nn.Module):
    def __init__(self, args, ind_to_classes):
        super(neural_motifs_sg2im_model, self).__init__()
        self.args = args

        # define and initial detector
        # self.detector = ObjectDetector(classes=ind_to_classes, num_gpus=args.num_gpus,
        #                             mode='refinerels' if not args.use_proposals else 'proposals',
        #                             use_resnet=args.use_resnet)
        # if args.ckpt is not None:
        #     ckpt = torch.load(args.ckpt)
        #     optimistic_restore(self.detector, ckpt['state_dict'])
        # self.detector.eval()

        # define and initial generator, image_discriminator, obj_discriminator,
        # and corresponding optimizer
        vocab = {
                'object_idx_to_name': ind_to_classes,
        }
        self.model, model_kwargs = build_model(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        self.obj_discriminator, d_obj_kwargs = build_obj_discriminator(args, vocab)
        self.img_discriminator, d_img_kwargs = build_img_discriminator(args)

        if self.obj_discriminator is not None:
            self.obj_discriminator.train()
            self.optimizer_d_obj = torch.optim.Adam(self.obj_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

        if self.img_discriminator is not None:
            self.img_discriminator.train()
            self.optimizer_d_img = torch.optim.Adam(self.img_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

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

            if self.obj_discriminator is not None:
                self.obj_discriminator.load_state_dict(checkpoint['d_obj_state'])
                self.optimizer_d_obj.load_state_dict(checkpoint['d_obj_optim_state'])

            if self.img_discriminator is not None:
                self.img_discriminator.load_state_dict(checkpoint['d_img_state'])
                self.optimizer_d_img.load_state_dict(checkpoint['d_img_optim_state'])

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

        self.t, self.epoch, self.checkpoint = t, epoch, checkpoint
        self.forward_G = True
        self.calc_G_D_loss = True
        self.forward_D = True

    def forward(self, imgs, img_offset, gt_boxes, gt_classes, gt_fmaps):
        # forward detector
        # with timeit('detector forward', self.args.timing):
        #     result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
        #                                train_anchor_inds, return_fmap=True)
        # if result.is_none():
        #     return ValueError("heck")

        # forward generator
        # imgs = F.interpolate(x, size=self.args.image_size)
        # objs = result.obj_preds
        # boxes = result.rm_box_priors / BOX_SCALE
        # obj_to_img = result.im_inds - image_offset
        # obj_fmap = result.obj_fmap
        #
        # # check if all image have detection
        # cnt = torch.zeros(len(imgs)).byte()
        # cnt[obj_to_img] += 1
        # if (cnt > 0).sum() != len(imgs):
        #     print("some imgs have no detection")
        #     print(cnt)
        #     imgs = imgs[cnt]
        #     obj_to_img_new = obj_to_img.clone()
        #     for i in range(len(cnt)):
        #         if cnt[i] == 0:
        #             obj_to_img_new -= (obj_to_img > i).long()
        #     obj_to_img = obj_to_img_new

        obj_to_img = gt_classes[:, 0] - img_offset
        # print("obj_to_img.min(), obj_to_img.max(), len(imgs) {} {} {}".format(obj_to_img.min(), obj_to_img.max(), len(imgs)))
        assert obj_to_img.min() >= 0 and obj_to_img.max() < len(imgs), \
            "obj_to_img.min() >= 0 and obj_to_img.max() < len(imgs) is not satidfied: {} {} {}".format(obj_to_img.min(), obj_to_img.max(), len(imgs))
        boxes = gt_boxes
        obj_fmap = gt_fmaps
        objs = gt_classes[:, 1]

        if self.forward_G:
            with timeit('generator forward', self.args.timing):
                imgs_pred = self.model(obj_to_img, boxes, obj_fmap)

        g_scores_fake_crop, g_obj_scores_fake_crop = None, None
        g_scores_fake_img = None
        if self.calc_G_D_loss:
            # forward discriminators to train generator
            if self.obj_discriminator is not None:
                with timeit('d_obj forward for g', self.args.timing):
                    g_scores_fake_crop, g_obj_scores_fake_crop = self.obj_discriminator(imgs_pred, objs, boxes, obj_to_img)

            if self.img_discriminator is not None:
                with timeit('d_img forward for g', self.args.timing):
                    g_scores_fake_img = self.img_discriminator(imgs_pred)

        d_scores_fake_crop, d_obj_scores_fake_crop = None, None
        d_scores_real_crop, d_obj_scores_real_crop = None, None
        d_obj_gp = None
        d_scores_fake_img = None
        d_scores_real_img = None
        d_img_gp = None
        if self.forward_D:
            # forward discriminators to train discriminators
            if self.obj_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_obj forward for d', self.args.timing):
                    d_scores_fake_crop, d_obj_scores_fake_crop, fake_crops = self.obj_discriminator(imgs_fake, objs, boxes, obj_to_img, return_crops=True)
                    d_scores_real_crop, d_obj_scores_real_crop, real_crops = self.obj_discriminator(imgs, objs, boxes, obj_to_img, return_crops=True)
                    if self.args.gan_loss_type == "wgan-gp":
                        d_obj_gp = gradient_penalty(real_crops, fake_crops, self.obj_discriminator)

            if self.img_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_img forward for d', self.args.timing):
                    d_scores_fake_img = self.img_discriminator(imgs_fake)
                    d_scores_real_img = self.img_discriminator(imgs)
                    if self.args.gan_loss_type == "wgan-gp":
                        d_img_gp = gradient_penalty(imgs, imgs_fake, self.img_discriminator)

        return Result(
            imgs=imgs,
            imgs_pred=imgs_pred,
            objs=objs,
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
            d_img_gp=d_img_gp
        )
        # return imgs, imgs_pred, objs, g_scores_fake_crop, g_obj_scores_fake_crop, g_scores_fake_img, d_scores_fake_crop, \
        #        d_obj_scores_fake_crop, d_scores_real_crop, d_obj_scores_real_crop, d_scores_fake_img, d_scores_real_img

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.args.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.args.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.args.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs

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
            d_img_gp=None):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([v is None for k, v in self.__dict__.items() if k != 'self'])
