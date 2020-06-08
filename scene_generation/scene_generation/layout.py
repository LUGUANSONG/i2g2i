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

import numpy as np
import torch
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
import torch.backends.cudnn as cudnn

"""
Functions for computing image layouts from object vectors, bounding boxes,
and segmentation masks. These are used to compute course scene layouts which
are then fed as input to the cascaded refinement network.
"""


def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - obj_to_img: LongTensor of shape (O,) mapping each element of vecs to
      an image, where each element is in the range [0, N). If obj_to_img[i] = j
      then vecs[i] belongs to image j.
    - H, W: Size of the output

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    O, D = vecs.size()
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    # If we don't add extra spatial dimensions here then out-of-bounds
    # elements won't be automatically set to 0
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    # Explicitly masking makes everything quite a bit slower.
    # If we rely on implicit masking the interpolated boxes end up
    # blurred around the edges, but it should be fine.
    # mask = ((X < 0) + (X > 1) + (Y < 0) + (Y > 1)).clamp(max=1)
    # sampled[mask[:, None]] = 0

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum', test_mode=False):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - masks: Tensor of shape (O, M, M) giving binary masks for each object
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - H, W: Size of the output image.

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.contiguous().view(O, D, 1, 1) * masks.contiguous().float().view(O, 1, M, M)
    # img_in = img_in.contiguous()
    # grid = grid.contiguous()
    # sampled = F.grid_sample(img_in, grid)
    sampled = GridSampler.apply(img_in, grid, 'zeros')
    if test_mode:
        clean_mask_sampled = F.grid_sample(masks.float().view(O, 1, M, M), grid)
    else:
        clean_mask_sampled = None

    out = _pool_samples(sampled, clean_mask_sampled, obj_to_img, pooling=pooling)
    return out


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2] - x0, boxes[:, 3] - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def _pool_samples(samples, clean_mask_sampled, obj_to_img, pooling='sum'):
    """
    Input:
    - samples: FloatTensor of shape (O, D, H, W)
    - obj_to_img: LongTensor of shape (O,) with each element in the range
      [0, N) mapping elements of samples to output images

    Output:
    - pooled: FloatTensor of shape (N, D, H, W)
    """
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    # Use scatter_add to sum the sampled outputs for each image
    # out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    # idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    # out = out.scatter_add(0, idx, samples)
    obj_to_img_list = [i.item() for i in list(obj_to_img)]
    all_out = []
    if clean_mask_sampled is None:
        for i in range(N):
            start = obj_to_img_list.index(i)
            end = len(obj_to_img_list) - obj_to_img_list[::-1].index(i)
            all_out.append(torch.sum(samples[start:end, :, :, :], dim=0))
    else:
        _, d, h, w = samples.shape
        for i in range(N):
            start = obj_to_img_list.index(i)
            end = len(obj_to_img_list) - obj_to_img_list[::-1].index(i)
            mass = [torch.sum(samples[j, :, :, :]).item() for j in range(start, end)]
            argsort = np.argsort(mass)
            result = torch.zeros((d, h, w), device=samples.device, dtype=samples.dtype)
            result_clean = torch.zeros((h, w), device=samples.device, dtype=samples.dtype)
            for j in argsort:
                masked_mask = (result_clean == 0).float() * (clean_mask_sampled[start + j, 0] > 0.5).float()
                result_clean += masked_mask
                result += samples[start + j] * masked_mask
            all_out.append(result)
    out = torch.stack(all_out)

    if pooling == 'avg':
        # Divide each output mask by the number of objects; use scatter_add again
        # to count the number of objects per image.
        ones = torch.ones(O, dtype=dtype, device=device)
        obj_counts = torch.zeros(N, dtype=dtype, device=device)
        obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
        # print(obj_counts)
        obj_counts = obj_counts.clamp(min=1)
        out = out / obj_counts.view(N, 1, 1, 1)
    elif pooling != 'sum':
        raise ValueError('Invalid pooling "%s"' % pooling)

    return out


MODE_ZEROS = 0
MODE_BORDER = 1

class GridSampler(Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode='zeros'):
        ctx.save_for_backward(input, grid)
        if padding_mode == 'zeros':
            ctx.padding_mode = MODE_ZEROS
        elif padding_mode == 'border':
            ctx.padding_mode = MODE_BORDER
        else:
            raise ValueError("padding_mode needs to be 'zeros' or 'border', but got {}".format(padding_mode))
        grid_sz = grid.size()
        backend = type2backend[input.type()]
        if input.dim() == 4:
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2])
            backend.SpatialGridSamplerBilinear_updateOutput(backend.library_state, input, grid,
                                                            output, ctx.padding_mode)
        elif input.dim() == 5:
            output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2], grid_sz[3])
            backend.VolumetricGridSamplerBilinear_updateOutput(backend.library_state, input, grid,
                                                               output, ctx.padding_mode)
        else:
            raise ValueError("input has to be 4d or 5d but got input of shape: {}".format(input.shape))
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        padding_mode = ctx.padding_mode
        backend = type2backend[input.type()]
        grad_input = input.new(input.size())
        grad_grid = grid.new(grid.size())
        if input.dim() == 4:
            backend.SpatialGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output, padding_mode)
        elif input.dim() == 5:
            backend.VolumetricGridSamplerBilinear_updateGradInput(
                backend.library_state, input, grad_input,
                grid, grad_grid, grad_output, padding_mode)
        else:
            raise ValueError("input has to be 4d or 5d but got input of shape: {}".format(input.shape))
        return grad_input, grad_grid, None



if __name__ == '__main__':
    vecs = torch.FloatTensor([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ])
    boxes = torch.FloatTensor([
        [0.25, 0.125, 0.5, 0.875],
        [0, 0, 1, 0.25],
        [0.6125, 0, 0.875, 1],
        [0, 0.8, 1, 1.0],
        [0.25, 0.125, 0.5, 0.875],
        [0.6125, 0, 0.875, 1],
    ])
    obj_to_img = torch.LongTensor([0, 0, 0, 1, 1, 1])
    # vecs = torch.FloatTensor([[[1]]])
    # boxes = torch.FloatTensor([[[0.25, 0.25, 0.75, 0.75]]])
    vecs, boxes = vecs.cuda(), boxes.cuda()
    obj_to_img = obj_to_img.cuda()
    out = boxes_to_layout(vecs, boxes, obj_to_img, 256, pooling='sum')

    from torchvision.utils import save_image

    save_image(out.data, 'out.png')

    masks = torch.FloatTensor([
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    ])
    masks = masks.cuda()
    out = masks_to_layout(vecs, boxes, masks, obj_to_img, 256)
    save_image(out.data, 'out_masks.png')
