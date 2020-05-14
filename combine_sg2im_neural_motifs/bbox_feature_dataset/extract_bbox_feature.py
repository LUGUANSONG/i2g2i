import load_detector
from tqdm import tqdm
import torch
import os
from os.path import join
import pickle


args = load_detector.conf

save_path = "./data/"

def get_bbox_feature(loader, str, fns_list, flipped_list, objs_list, fmap_list, bbox_list):
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
                fns_list.append(img_fns[i])
                flipped_list.append(flipped[i])
                objs_list.append(objs[ind].cpu())
                fmap_list.append(obj_fmap[ind].cpu())
                bbox_list.append(boxes[ind].cpu())
            else:
                print("no bbox detected in %s" % img_fns[i])


val_fns = []
val_flipped = []
val_objs = []
val_fmap = []
val_bbox = []
get_bbox_feature(load_detector.val_loader, "Val Loader", val_fns, val_flipped, val_objs,
                 val_fmap, val_bbox)
val_dataset = {
    'fns': val_fns,
    'flipped': val_flipped,
    'objs': val_objs,
    'fmap': val_fmap,
    'bbox': val_bbox
}
print(len(val_fns))
print(len(val_flipped))
print(len(val_objs))
print(len(val_fmap))
print(len(val_bbox))
print("dumping vg_val_bbox_feature.pkl")
pickle.dump(val_dataset, open(join(save_path, "vg_val_bbox_feature.pkl"), "wb"), protocol=4)
print("finish")


train_fns = []
train_flipped = []
train_objs = []
train_fmap = []
train_bbox = []
get_bbox_feature(load_detector.train_not_flip_loader, "Train Not Flip Loader", train_fns, train_flipped, train_objs,
                 train_fmap, train_bbox)
get_bbox_feature(load_detector.train_flip_loader, "Train Flip Loader", train_fns, train_flipped, train_objs,
                 train_fmap, train_bbox)
train_dataset = {
    'fns': train_fns,
    'flipped': train_flipped,
    'objs': train_objs,
    'fmap': train_fmap,
    'bbox': train_bbox
}
print(len(train_fns))
print(len(train_flipped))
print(len(train_objs))
print(len(train_fmap))
print(len(train_bbox))
print("dumping vg_train_bbox_feature.pkl")
pickle.dump(train_dataset, open(join(save_path, "vg_train_bbox_feature.pkl"), "wb"), protocol=4)
print("finish")


test_fns = []
test_flipped = []
test_objs = []
test_fmap = []
test_bbox = []
get_bbox_feature(load_detector.test_loader, "Test Loader", test_fns, test_flipped, test_objs,
                 test_fmap, test_bbox)
test_dataset = {
    'fns': test_fns,
    'flipped': test_flipped,
    'objs': test_objs,
    'fmap': test_fmap,
    'bbox': test_bbox
}
print(len(test_fns))
print(len(test_flipped))
print(len(test_objs))
print(len(test_fmap))
print(len(test_bbox))
pickle.dump(test_dataset, open(join(save_path, "vg_test_bbox_feature.pkl"), "wb"), protocol=4)
