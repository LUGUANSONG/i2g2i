import load_detector
from tqdm import tqdm
import torch
import os
from os.path import join, dirname
import pickle
from config import VG_IMAGES


args = load_detector.conf

# save_path = "./data/"
save_path = dirname(VG_IMAGES)
global_index = 0

def get_bbox_feature(loader, str, save_path):
    global global_index
    for step, batch in enumerate(tqdm(loader, desc=str, total=len(loader))):
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

        # print(len(img_fns))
        # print(len(flipped))
        # print(objs.shape)
        # print(boxes.shape)
        # print(obj_to_img.shape)
        # print(obj_fmap.shape)

        for i in range(len(img_fns)):
            ind = (obj_to_img == i).nonzero()
            if len(ind) > 0:
                ind = ind.squeeze(1)
                # fns_list.append(img_fns[i])
                # flipped_list.append(flipped[i])
                # objs_list.append(objs[ind].cpu())
                # fmap_list.append(obj_fmap[ind].cpu())
                # bbox_list.append(boxes[ind].cpu())
                save_dict = {
                    'fns': img_fns[i],
                    'flipped': flipped[i],
                    'objs': objs[ind].cpu(),
                    'fmap': obj_fmap[ind].cpu(),
                    'bbox': boxes[ind].cpu()
                }
                pickle.dump(save_dict, open(join(save_path, "%d.pkl" % global_index), "wb"), protocol=4)
                global_index += 1
            else:
                print("no bbox detected in %s" % img_fns[i])

global_index = 0
get_bbox_feature(load_detector.val_loader, "Val Loader", join(save_path, "val"))
print("finish val dataset")

global_index = 0
get_bbox_feature(load_detector.train_not_flip_loader, "Train Not Flip Loader", join(save_path, "train"))
get_bbox_feature(load_detector.train_flip_loader, "Train Flip Loader", join(save_path, "train"))
print("finish train dataset")

global_index = 0
get_bbox_feature(load_detector.test_loader, "Test Loader", join(save_path, "test"))
print("finish test dataset")
