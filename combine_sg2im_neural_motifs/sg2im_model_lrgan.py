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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sg2im.box_utils as box_utils
from sg2im.graph import GraphTripleConv, GraphTripleConvNet
from sg2im.crn import RefinementNetwork
from sg2im.layout import boxes_to_layout, masks_to_layout
from sg2im.layers import build_mlp


class Sg2ImModel(nn.Module):
  def __init__(self, image_size=(64, 64), gconv_dim=128,
               refinement_dims=(1024, 512, 256, 128, 64),
               normalization='batch', activation='leakyrelu-0.2',
               mask_size=None, layout_noise_dim=0, args=None,
               **kwargs):
    super(Sg2ImModel, self).__init__()
    print("i2g2i.combine_sg2im_neural_motifs.sg2im_model.Sg2ImModel")

    # We used to have some additional arguments: 
    # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
    if len(kwargs) > 0:
      print('WARNING: Model got unexpected kwargs ', kwargs)

    # self.vocab = vocab
    self.image_size = image_size
    self.layout_noise_dim = layout_noise_dim
    self.gconv_dim = gconv_dim
    self.args = args

    # num_objs = len(vocab['object_idx_to_name'])
    # num_preds = len(vocab['pred_idx_to_name'])
    # self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
    # self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

    # if gconv_num_layers == 0:
    #   self.gconv = nn.Linear(embedding_dim, gconv_dim)
    # elif gconv_num_layers > 0:
    #   gconv_kwargs = {
    #     'input_dim': embedding_dim,
    #     'output_dim': gconv_dim,
    #     'hidden_dim': gconv_hidden_dim,
    #     'pooling': gconv_pooling,
    #     'mlp_normalization': mlp_normalization,
    #   }
    #   self.gconv = GraphTripleConv(**gconv_kwargs)

    # self.gconv_net = None
    # if gconv_num_layers > 1:
    #   gconv_kwargs = {
    #     'input_dim': gconv_dim,
    #     'hidden_dim': gconv_hidden_dim,
    #     'pooling': gconv_pooling,
    #     'num_layers': gconv_num_layers - 1,
    #     'mlp_normalization': mlp_normalization,
    #   }
    #   self.gconv_net = GraphTripleConvNet(**gconv_kwargs)
    self.obj_fmap_net = nn.Linear(4096, gconv_dim)

    # box_net_dim = 4
    # box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
    # self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)

    if args.noise_apply_method == "concat":
      factor = 1
    else:
      raise Exception("unrecognized noise_apply_method: %s" % args.noise_apply_method)

    self.mask_net = None
    object_factor = 1
    if mask_size is not None and mask_size > 0:
      self.mask_net = self._build_mask_net(gconv_dim + args.object_noise_dim * factor, mask_size)
      if args.object_no_noise_with_mask:
        object_factor = 0
    else:
      if args.object_no_noise_with_bbox:
        object_factor = 0

    # rel_aux_layers = [2 * embedding_dim + 8, gconv_hidden_dim, num_preds]
    # self.rel_aux_net = build_mlp(rel_aux_layers, batch_norm=mlp_normalization)

    refinement_kwargs = {
      'dims': (gconv_dim + args.object_noise_dim * factor * object_factor + layout_noise_dim * factor,) + refinement_dims,
      'normalization': normalization,
      'activation': activation,
    }
    self.refinement_net = RefinementNetwork(**refinement_kwargs)

  def _build_mask_net(self, dim, mask_size):
    output_dim = 1
    layers, cur_size = [], 1
    while cur_size < mask_size:
      layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
      layers.append(nn.BatchNorm2d(dim))
      layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
      layers.append(nn.ReLU())
      cur_size *= 2
    if cur_size != mask_size:
      raise ValueError('Mask size must be a power of 2')
    layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
    return nn.Sequential(*layers)

  def forward(self, obj_to_img, boxes_gt, obj_fmaps, mask_noise_indexes=None, masks_gt=None):
    """
    Required Inputs:
    - objs: LongTensor of shape (O,) giving categories for all objects
    - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
      means that there is a triple (objs[s], p, objs[o])

    Optional Inputs:
    - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
      means that objects[o] is an object in image i. If not given then
      all objects are assumed to belong to the same image.
    - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
      the spatial layout; if not given then use predicted boxes.
    """
    assert boxes_gt.max() < 1.1 and boxes_gt.min() > -0.1, "boxes_gt should be within range [0,1]"
    # O, T = objs.size(0), triples.size(0)
    # s, p, o = triples.chunk(3, dim=1)           # All have shape (T, 1)
    # s, p, o = [x.squeeze(1) for x in [s, p, o]] # Now have shape (T,)
    # edges = torch.stack([s, o], dim=1)          # Shape is (T, 2)
    #
    # if obj_to_img is None:
    #   obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)
    #
    # obj_vecs = self.obj_embeddings(objs)
    # obj_vecs_orig = obj_vecs
    # pred_vecs = self.pred_embeddings(p)
    #
    # if isinstance(self.gconv, nn.Linear):
    #   obj_vecs = self.gconv(obj_vecs)
    # else:
    #   obj_vecs, pred_vecs = self.gconv(obj_vecs, pred_vecs, edges)
    # if self.gconv_net is not None:
    #   obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)
    obj_vecs = self.obj_fmap_net(obj_fmaps)
    no_noise_obj_vecs = obj_vecs
    if self.args.object_noise_dim > 0:
      # select objs belongs to images in mask_noise_indexes
      if mask_noise_indexes is not None and self.training:
        mask_noise_obj_index_list = []
        for ind in mask_noise_indexes:
          mask_noise_obj_index_list.append((obj_to_img == ind).nonzero())
        mask_noise_obj_indexes = torch.cat(mask_noise_obj_index_list, dim=0)[:, 0]

      object_noise = torch.randn((obj_vecs.shape[0], self.args.object_noise_dim), dtype=obj_vecs.dtype,
                                 device=obj_vecs.device)
      if mask_noise_indexes is not None and self.training:
        object_noise[mask_noise_obj_indexes] = 0
      obj_vecs = torch.cat([obj_vecs, object_noise], dim=1)

    # boxes_pred = self.box_net(obj_vecs)

    masks_pred = None
    if self.mask_net is not None:
      mask_scores = self.mask_net(obj_vecs.view(obj_vecs.shape[0], -1, 1, 1))
      masks_pred = mask_scores.squeeze(1).sigmoid()

    # s_boxes, o_boxes = boxes_pred[s], boxes_pred[o]
    # s_vecs, o_vecs = obj_vecs_orig[s], obj_vecs_orig[o]
    # rel_aux_input = torch.cat([s_boxes, o_boxes, s_vecs, o_vecs], dim=1)
    # rel_scores = self.rel_aux_net(rel_aux_input)

    H, W = self.image_size
    # layout_boxes = boxes_pred if boxes_gt is None else boxes_gt
    layout_boxes = boxes_gt

    # layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)

    if masks_pred is None:
      if self.args.object_no_noise_with_bbox:
        layout = boxes_to_layout(no_noise_obj_vecs, layout_boxes, obj_to_img, H, W)
      else:
        layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
    else:
      layout_masks = masks_pred if masks_gt is None else masks_gt
      if self.args.object_no_noise_with_mask:
        layout = masks_to_layout(no_noise_obj_vecs, layout_boxes, layout_masks,
                                 obj_to_img, H, W)
      else:
        layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                 obj_to_img, H, W)
    ret_layout = layout

    if self.layout_noise_dim > 0:
      N, C, H, W = layout.size()
      noise_shape = (N, self.layout_noise_dim, H, W)
      # print("check noise_std here, it is %.10f" % self.args.noise_std)
      noise_std = torch.zeros(noise_shape, dtype=layout.dtype,
                                 device=layout.device).fill_(self.args.noise_std)
      layout_noise = torch.normal(mean=0.0, std=noise_std)
      if mask_noise_indexes is not None and self.training:
        layout_noise[mask_noise_indexes] = 0.
      # layout_noise = torch.randn(noise_shape, dtype=layout.dtype,
      #                            device=layout.device)
      layout = torch.cat([layout, layout_noise], dim=1)

    img = self.refinement_net(layout)
    return img, ret_layout, layout[:, self.gconv_dim:, :, :]
