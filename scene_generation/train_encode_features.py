#!/usr/bin/python
import os
import sys
import json
from collections import defaultdict
import torch

from scene_generation.args import get_args
from combine_sg2im_neural_motifs.bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
from torchvision import transforms
from scene_generation.trainer import Trainer
from scene_generation.data import imagenet_deprocess_batch
from scene_generation.bilinear import crop_bbox_batch
from scipy.misc import imsave
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def cluster(features, num_objs, n_clusters, save_path):
    name = 'features'
    centers = {}
    for label in range(num_objs):
        feat = features[label]
        if feat.shape[0]:
            n_feat_clusters = min(feat.shape[0], n_clusters)
            if n_feat_clusters < n_clusters:
                print(label)
            kmeans = KMeans(n_clusters=n_feat_clusters, random_state=0).fit(feat)
            if n_feat_clusters == 1:
                centers[label] = kmeans.cluster_centers_
            else:
                one_dimension_centers = TSNE(n_components=1).fit_transform(kmeans.cluster_centers_)
                args = np.argsort(one_dimension_centers.reshape(-1))
                centers[label] = kmeans.cluster_centers_[args]
    save_name = os.path.join(save_path, name + '_clustered_%03d.npy' % n_clusters)
    np.save(save_name, centers)
    print('saving to %s' % save_name)


def check_model(args, loader, model, checkpoint):
    model.forward_D = False
    model.eval()

    name = 'features'
    rep_size = checkpoint['model_kwargs']['rep_size']
    vocab = checkpoint['model_kwargs']['vocab']
    num_objs = len(vocab['object_to_idx'])

    save_path = args.output_dir

    ########### Encode features ###########
    counter = 0
    max_counter = 1000000000
    if not args.only_clustering:
        print('begin')
        with torch.no_grad():
            features = {}
            crops_dict = {}
            for label in range(num_objs):
                features[label] = np.zeros((0, rep_size))
                crops_dict[label] = []
            # for i, batch in enumerate(loader):
            for t, batch in enumerate(tqdm(loader, desc='extract object representation', total=len(loader))):
                if counter >= max_counter:
                    break
                # (all_imgs, all_objs, all_boxes, all_masks, all_triples,
                #            all_obj_to_img, all_triple_to_img, all_attributes)
                # imgs = data[0].cuda()
                # objs = data[1]
                # objs = [j.item() for j in objs]
                # boxes = data[2].cuda()
                # obj_to_img = data[5].cuda()
                # crops = crop_bbox_batch(imgs, boxes, obj_to_img, model.object_size)
                # feat = model.repr_net(model.image_encoder(crops)).cpu()

                result = model[batch]
                feat = result.obj_repr.cpu()
                objs = result.objs
                objs = [j.item() for j in objs]

                crops = result.crops.cpu()
                for ind, label in enumerate(objs):
                    features[label] = np.append(features[label], feat[ind].view(1, -1), axis=0)
                    crops_dict[label].append(crops[ind])

                counter += len(objs)

                min_size = -1
                max_size = -1
                for k, v in features.items():
                    if min_size == -1:
                        min_size = v.shape[0]
                    else:
                        min_size = min(min_size, v.shape[0])

                    if max_size == -1:
                        max_size = v.shape[0]
                    else:
                        max_size = max(max_size, v.shape[0])
                print("min_size: %d, max_size: %d" % (min_size, max_size))
                if args.num_val_samples > 0 and min_size >= args.num_val_samples:
                    break

                # print('%d / %d images' % (i + 1, dataset_size))
            save_name = os.path.join(save_path, name + '.npy')
            np.save(save_name, features)
            print("finish save %s" % save_name)
            byte_size = sys.getsizeof(crops_dict)
            print("size of crops_dict: %dB, %.3fMB" % (byte_size, byte_size / 1024. / 1024.))
            pickle.dump(crops_dict, open(os.path.join(save_path, name + "_crops.pkl"), "wb"), protocol=4)
            print("finish save %s" % os.path.join(save_path, name + "_crops.pkl"))

    if not args.not_clustering:
        ############## Clustering ###########
        print('begin clustering')
        load_name = os.path.join(save_path, name + '.npy')
        features = np.load(load_name, allow_pickle=True).item()
        cluster(features, num_objs, 100, save_path)
        cluster(features, num_objs, 10, save_path)
        cluster(features, num_objs, 1, save_path)


def get_checkpoint(args, vocab):
    if args.restore_from_checkpoint:
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(args.output_dir, restore_path)
        assert restore_path is not None
        assert os.path.isfile(restore_path)
        print('Restoring from checkpoint:')
        print(restore_path)
        checkpoint = torch.load(restore_path)
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
    else:
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'vocab': vocab,
            'model_kwargs': {},
            'd_obj_kwargs': {},
            'd_mask_kwargs': {},
            'd_img_kwargs': {},
            'd_global_mask_kwargs': {},
            'losses_ts': [],
            'losses': defaultdict(list),
            'd_losses': defaultdict(list),
            'checkpoint_ts': [],
            'train_inception': [],
            'val_losses': defaultdict(list),
            'val_inception': [],
            'norm_d': [],
            'norm_g': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'model_state': None, 'model_best_state': None,
            'optim_state': None, 'optim_best_state': None,
            'd_obj_state': None, 'd_obj_best_state': None,
            'd_obj_optim_state': None, 'd_obj_optim_best_state': None,
            'd_img_state': None, 'd_img_best_state': None,
            'd_img_optim_state': None, 'd_img_optim_best_state': None,
            'd_mask_state': None, 'd_mask_best_state': None,
            'd_mask_optim_state': None, 'd_mask_optim_best_state': None,
            'best_t': [],
        }
    return t, epoch, checkpoint


def main(args):
    print(args)
    train, val, test = VG.splits(transform=transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]))
    vocab = {
        'object_to_idx': {train.ind_to_classes[i]: (i - 1) for i in range(1, len(train.ind_to_classes))},
    }
    train_loader, test_loader = VGDataLoader.splits(train, test, batch_size=args.batch_size,
                                                   num_workers=args.loader_num_workers,
                                                   num_gpus=args.num_gpus)
    print(train.ind_to_classes)

    t, epoch, checkpoint = get_checkpoint(args, vocab)
    trainer = Trainer(args, vocab, checkpoint)
    print(trainer)
    if args.restore_from_checkpoint:
        trainer.restore_checkpoint(checkpoint)
    else:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as outfile:
            json.dump(vars(args), outfile)

    print('checking on test')
    print("t: %d, epoch: %d" % (t, epoch))
    with open(os.path.join(args.output_dir, "test_metrics.txt"), "a") as f:
        f.write("t: %d, epoch: %d\n\n" % (t, epoch))
    check_model(args, train_loader, trainer, checkpoint)


if __name__ == '__main__':
    args = get_args()
    main(args)
