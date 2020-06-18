import argparse
import os
from random import randint

import numpy as np
import torch
from scipy.misc import imsave

from combine_sg2im_neural_motifs.bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
from torchvision import transforms

from scene_generation.data import imagenet_deprocess_batch
from scene_generation.model import Model
from scene_generation.utils import int_tuple, bool_flag


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])
parser.add_argument('--accuracy_model_path', default=None)

# Shared dataset options
parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=10000, type=int)
parser.add_argument('--save_gt_imgs', default=False, type=bool_flag)
parser.add_argument('--use_gt_textures', default=False, type=bool_flag)
parser.add_argument('--output_dir', default='output')
parser.add_argument('--test_on_train', default=False, type=bool_flag)


def build_model(args, checkpoint):
    kwargs = checkpoint['model_kwargs']
    model = Model(**kwargs)
    model_state = checkpoint['model_state']
    model.load_state_dict(model_state)
    if args.model_mode == 'eval':
        model.eval()
    elif args.model_mode == 'train':
        model.train()
    model.image_size = args.image_size
    model.cuda()
    return model


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def one_hot_to_rgb(layout_pred, colors, num_objs):
    one_hot = layout_pred[:, :num_objs, :, :]
    one_hot_3d = torch.einsum('abcd,be->aecd', [one_hot.cpu(), colors])
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def run_model(args, checkpoint, output_dir, loader=None):
    dirname = os.path.dirname(args.checkpoint)
    features = None
    if not args.use_gt_textures:
        features_path = os.path.join(dirname, 'features_clustered_001.npy')
        print(features_path)
        if os.path.isfile(features_path):
            features = np.load(features_path, allow_pickle=True).item()
        else:
            raise ValueError('No features file')
    with torch.no_grad():
        vocab = checkpoint['model_kwargs']['vocab']
        model = build_model(args, checkpoint)
        if loader is None:
            train, val, test = VG.splits(transform=transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]))
            train_loader, test_loader = VGDataLoader.splits(train, test, batch_size=args.batch_size,
                                                           num_workers=args.loader_num_workers,
                                                           num_gpus=1)
            if args.test_on_train:
                loader = train_loader
            else:
                loader = test_loader
            print(train.ind_to_classes)

        img_dir = makedir(output_dir, 'test')
        gt_img_dir = makedir(output_dir, 'test_real', args.save_gt_imgs)

        img_idx = 0
        for batch in loader:
            batch.scatter()
            assert not isinstance(batch.imgs, list), "single gpu, batch should contain only one partition, so should no be a list, but a tensor directly"
            gt_imgs, img_offset, boxes_gt, gt_classes, gt_fmaps = batch[0]
            assert img_offset == 0, "single gpu, img_offset should be 0"

            objs = gt_classes[:, 1]
            obj_to_img = gt_classes[:, 0]

            if args.use_gt_textures:
                all_features = None
            else:
                all_features = []
                for obj_name in objs:
                    obj_feature = features[obj_name.item()]
                    random_index = randint(0, obj_feature.shape[0] - 1)
                    feat = torch.from_numpy(obj_feature[random_index, :]).type(torch.float32).cuda()
                    all_features.append(feat)

            imgs_pred = model(gt_imgs, objs, gt_fmaps, obj_to_img, boxes_gt=boxes_gt, test_mode=True,
                              use_gt_box=True, features=all_features)[0]
            print(gt_imgs.shape, img_offset, boxes_gt.shape, gt_classes.shape, gt_fmaps.shape, imgs_pred.shape)

            imgs_gt = imagenet_deprocess_batch(gt_imgs)
            imgs_pred = imagenet_deprocess_batch(imgs_pred)
            for i in range(imgs_pred.size(0)):
                img_filename = '%04d.png' % img_idx
                if args.save_gt_imgs:
                    img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)
                    img_gt_path = os.path.join(gt_img_dir, img_filename)
                    imsave(img_gt_path, img_gt)

                img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(img_dir, img_filename)
                imsave(img_path, img_pred_np)

                img_idx += 1

            print('Saved %d images' % img_idx)
            if img_idx >= args.num_samples:
                break


if __name__ == '__main__':
    args = parser.parse_args()
    if args.checkpoint is None:
        raise ValueError('Must specify --checkpoint')

    checkpoint = torch.load(args.checkpoint)
    print('Loading model from ', args.checkpoint)
    run_model(args, checkpoint, args.output_dir)
