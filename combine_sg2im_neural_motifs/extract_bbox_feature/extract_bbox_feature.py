import .load_detector
from tqdm import tqdm
import torch
import os


args = load_detector.conf

train_fns = []
train_flipped = []
train_objs = []
train_fmap = []
train_bbox = []

for step, batch in enumerate(tqdm(load_detector.train_not_flip_loader, desc='Train Not Flip Loader',
                                  total=len(load_detector.train_loader))):
    with torch.no_grad():
        result = load_detector.detector[batch]
    imgs = batch.imgs
    img_fns = batch.fns
    flipped = batch.flipped

    objs = result.obj_preds
    boxes = result.rm_box_priors
    obj_to_img = result.im_inds
    obj_fmap = result.obj_fmap
    boxes /= load_detector.IM_SCALE

    for i in range(len(imgs)):
        ind = (obj_to_img == i).nonzero().squeeze(1)
        if len(ind) > 0:
            train_fns.append(img_fns[i])
            train_flipped.append(flipped[i])
            train_objs.append(objs[ind].cpu())
            train_fmap.append(obj_fmap[ind].cpu())
            train_bbox.append(boxes[ind].cpu())
            print(train_fns[-1])
            print(train_flipped[-1])
            print(train_objs[-1].shape)
            print(train_fmap[-1].shape)
            print(train_bbox[-1].shape)
    os._exit(0)
