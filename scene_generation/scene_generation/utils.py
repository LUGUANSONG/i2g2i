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

import random

import torch


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)


class LossManager(object):
    def __init__(self):
        self.total_loss = None
        self.all_losses = {}

    def add_loss(self, loss, name, weight=1.0, use_loss=True):
        cur_loss = loss * weight
        if use_loss:
            if self.total_loss is not None:
                self.total_loss += cur_loss
            else:
                self.total_loss = cur_loss

        self.all_losses[name] = cur_loss.data.cpu().item()

    def items(self):
        return self.all_losses.items()


class VectorPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.vectors = {}

    def query(self, objs, vectors):
        if self.pool_size == 0:
            return vectors
        return_vectors = []
        for obj, vector in zip(objs, vectors):
            obj = obj.item()
            vector = vector.cpu().clone().detach()
            if obj not in self.vectors:
                self.vectors[obj] = []
            obj_pool_size = len(self.vectors[obj])
            if obj_pool_size == 0:
                return_vectors.append(vector)
                self.vectors[obj].append(vector)
            elif obj_pool_size < self.pool_size:
                random_id = random.randint(0, obj_pool_size - 1)
                self.vectors[obj].append(vector)
                return_vectors.append(self.vectors[obj][random_id])
            else:
                random_id = random.randint(0, obj_pool_size - 1)
                tmp = self.vectors[obj][random_id]
                self.vectors[obj][random_id] = vector
                return_vectors.append(tmp)
        return_vectors = torch.stack(return_vectors).to(vectors.device)
        return return_vectors


class Result(object):
    def __init__(self,
                 imgs=None, imgs_pred=None, layout_pred=None,
                 scores_fake=None, ac_loss=None, mask_loss=None, loss_mask_feat=None, g_gan_img_loss=None,
                 loss_g_gan_feat_img=None, d_obj_gan_loss=None, ac_loss_real=None, ac_loss_fake=None, fake_loss=None,
                 real_loss=None, loss_d_fake_img=None, loss_d_wrong_texture=None, loss_D_real=None,
                 d_fake_crops=None, d_real_crops=None, obj_repr=None, objs=None, crops=None,
                 change_indexes=None, crop_indexes=None, boxes=None, obj_to_img=None
                 ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all([v is None for k, v in self.__dict__.items() if k != 'self'])
