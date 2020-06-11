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
import copy

# sg2im
from sg2im.losses import get_gan_losses
from sg2im.utils import timeit, bool_flag, LossManager

# neural motifs
# from dataloaders.visual_genome import VGDataLoader, VG
# from dataloaders.mscoco import CocoDetection, CocoDataLoader
from torchvision import transforms
from bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
# from config import ModelConfig
from config_args import config_args
from PIL import Image, ImageDraw, ImageFont

# combine
from model_bbox_feature import neural_motifs_sg2im_model

torch.backends.cudnn.benchmark = True

font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 8)


def draw_box(draw, boxx, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=2)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=2)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=2)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=2)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
        h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw


def check_args(args):
    H, W = args.image_size
    for _ in args.refinement_network_dims[1:]:
        H = H // 2
    if H == 0:
        raise ValueError("Too many layers in refinement network")


def check_model(args, loader, model, output_path):
    if not exists(output_path):
        os.makedirs(output_path)
    model.eval()
    num_samples = 0
    all_losses = defaultdict(list)
    model.forward_G = True
    model.calc_G_D_loss = False
    model.forward_D = False
    with torch.no_grad():
        for _batch in loader:
            for i in range(args.num_diff_noise):
                if i == 0:
                    args.exchange_feat_cls = False
                else:
                    args.exchange_feat_cls = True

                batch = copy.deepcopy(_batch)
                result = model[batch]
                imgs, imgs_pred, objs = result.imgs, result.imgs_pred, result.objs
                boxes, obj_to_img = result.boxes, result.obj_to_img

                imgs_pred = imgs_pred.cpu()
                images = imgs_pred * torch.tensor([0.229, 0.224, 0.225], device=imgs_pred.device).reshape(1, 3, 1, 1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
                images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
                images = images - images_min
                images = images / (images_max - images_min)
                imgs_pred = images.clamp(min=0, max=1)

                imgs = imgs.cpu()
                images = imgs * torch.tensor([0.229, 0.224, 0.225], device=imgs.device).reshape(1, 3, 1, 1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
                images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
                images = images - images_min
                images = images / (images_max - images_min)
                imgs = images.clamp(min=0, max=1)

                for k, image in enumerate(imgs_pred):
                    out_dir = join(output_path, "%d" % (num_samples + k))
                    if not exists(out_dir):
                        os.makedirs(out_dir)

                    image = transforms.ToPILImage()(image).convert("RGB")
                    image.save(join(out_dir, "img_pred_%d.png" % i))

                    image = transforms.ToPILImage()(imgs[k]).convert("RGB")
                    image.save(join(out_dir, "img_%d.png" % i))

                    # draw bbox and class
                    image = torch.ones(256, 256)
                    image = transforms.ToPILImage()(image).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    index = (obj_to_img == k).nonzero()[:, 0]
                    for ind in index:
                        box = boxes[ind] * 256
                        cls = objs[ind] + 1
                        draw = draw_box(draw, box, loader.dataset.ind_to_classes[cls])
                    image.save(join(out_dir, "img_layout_%d.png" % i))

            num_samples += imgs.size(0)
            if num_samples >= args.num_val_samples:
                break


def main(args):
    print(args)
    check_args(args)
    if not exists(args.output_dir):
        os.makedirs(args.output_dir)

    train, val, test = VG.splits(transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ]), args=args)
    val = test
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   num_gpus=args.num_gpus)
    print(train.ind_to_classes)

    all_in_one_model = neural_motifs_sg2im_model(args, train.ind_to_classes)
    restore_path = '%s_with_model.pt' % args.checkpoint_name
    restore_path = os.path.join(args.output_dir, restore_path)
    print('Restoring from checkpoint:')
    print(restore_path)
    checkpoint = torch.load(restore_path)
    all_in_one_model.model.load_state_dict(checkpoint['model_state'])

    all_in_one_model.cuda()

    print('checking on test')
    check_model(args, val_loader, all_in_one_model, join(args.output_dir, args.output_subdir))

if __name__ == '__main__':
    args = config_args
    main(args)

