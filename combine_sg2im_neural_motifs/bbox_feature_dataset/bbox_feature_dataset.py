"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os
import sys
from os.path import join, basename, dirname, exists

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from .bbox_feature_blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from collections import defaultdict
import pickle
import time
import gc


class VG(Dataset):
    def __init__(self, mode, transform, dict_file=VG_SGG_DICT_FN):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """

        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode

        pickle_files_dir = join(dirname(VG_IMAGES), mode)
        pickle_files = list(filter(lambda x: x.endswith(".pkl"), os.listdir(pickle_files_dir)))
        self.pickle_files = [join(pickle_files_dir, file) for file in pickle_files]
        print("number of samples under %s: %d" % (pickle_files_dir, len(self.pickle_files)))

        # file_path = "data/vg_%s_bbox_feature.pkl" % mode
        # start_time = time.time()
        # print("start to read pickle file: %s" % file_path)
        # f = open(file_path, "rb")
        # gc.disable()
        # pickle_file = pickle.load(f)
        # gc.enable()
        # f.close()
        # print("take %.3fs to load pickle file: %s" % (time.time() - start_time, file_path))
        # self.filenames = pickle_file['fns']
        # self.flipped = pickle_file['flipped']
        # self.gt_classes = pickle_file['objs']
        # self.fmaps = pickle_file['fmap']
        # self.gt_boxes = pickle_file['bbox']

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

        # tform = [
        #     # SquarePad(),
        #     Resize((IM_SCALE, IM_SCALE)),
        #     ToTensor(),
        #     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ]
        # self.transform_pipeline = Compose(tform)
        self.transform = transform

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):
        '''
        train_dataset = {
            'fns': train_fns,
            'flipped': train_flipped,
            'objs': train_objs,
            'fmap': train_fmap,
            'bbox': train_bbox
        }
        '''
        pickle_file = pickle.load(open(self.pickle_files[index], "rb"))
        fn = pickle_file['fns']
        image_unpadded = Image.open(fn).convert('RGB')

        flipped = pickle_file['flipped']
        gt_boxes = pickle_file['bbox'].numpy()

        if flipped:
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)

        entry = {
            'img': self.transform(image_unpadded),
            'gt_boxes': gt_boxes,
            'gt_classes': pickle_file['objs'].numpy(),
            'index': index,
            'flipped': flipped,
            'fn': fn,
            'fmap': pickle_file['fmap']
        }

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.pickle_files)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    if isinstane(entry['img'], torch.Tensor):
        im_size = tuple(entry['img'].size())
        if len(im_size) != 3:
            raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load

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

if __name__ == "__main__":
    dataset = VG(
        mode='train',
        transform=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    )

    output_path = sys.argv[1]
    print("output to %s" % output_path)
    if not exists(output_path):
        os.makedirs(output_path)

    for i in range(10):
        entry = dataset[i]
        for k, v in entry.items():
            try:
                print(k, type(v), v.shape, v.dtype)
            except:
                print(k, type(v))

        draw = ImageDraw.Draw(entry['img'])
        for bbox, cls in zip(entry['gt_boxes'], entry['gt_classes']):
            gt_box = bbox.numpy() * 256
            draw = draw_box(draw, gt_box, dataset.ind_to_classes[cls])
        entry['img'].save(join(output_path, "%d.png" % i))

