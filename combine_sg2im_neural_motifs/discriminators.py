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

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2im.bilinear import crop_bbox_batch
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn


class PatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='batch', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0, args=None):
    super(PatchDiscriminator, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.discriminator.PatchDiscriminator")
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.down_to_1channel = args.down_to_1channel
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1)
    self.rec_feature = nn.Sequential(GlobalAvgPool(), nn.Linear(output_dim, 4096)) if args.d_img_rec_feat_weight > 0 else None

  def forward(self, x, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    feature = self.cnn(x)
    real_scores = self.classifier(feature) if self.down_to_1channel else feature
    if self.rec_feature is not None:
      return real_scores, self.rec_feature(feature)
    else:
      return real_scores


class AcDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               padding='same', pooling='avg'):
    super(AcDiscriminator, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.discriminator.AcDiscriminator")
    self.vocab = vocab

    cnn_kwargs = {
      'arch': arch,
      'normalization': normalization,
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = nn.Sequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name']) - 1

    self.real_classifier = nn.Linear(1024, 1)
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y=None):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    # ac_loss = F.cross_entropy(obj_scores, y)
    # return real_scores, ac_loss
    return real_scores, obj_scores


class AcCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='none', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(AcCropDiscriminator, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.discriminator.AcCropDiscriminator")
    self.vocab = vocab
    self.discriminator = AcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img, return_crops=False):
    crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
    # real_scores, ac_loss = self.discriminator(crops, objs)
    # return real_scores, ac_loss
    real_scores, obj_scores = self.discriminator(crops, objs)
    if return_crops:
      return real_scores, obj_scores, crops
    else:
      return real_scores, obj_scores
