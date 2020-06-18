import argparse
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from combine_sg2im_neural_motifs.bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
from torchvision import transforms

from scene_generation.bilinear import crop_bbox_batch
from scene_generation.model import Model
from scene_generation.utils import int_tuple, bool_flag

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--model_mode', default='eval', choices=['train', 'eval'])

# Shared dataset options
parser.add_argument('--image_size', default=(128, 128), type=int_tuple)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--shuffle', default=False, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--object_size', default=64, type=int)


def build_model(args, checkpoint):
    kwargs = checkpoint['model_kwargs']
    model = Model(**kwargs)
    model.load_state_dict(checkpoint['model_state'])
    if args.model_mode == 'eval':
        model.eval()
    elif args.model_mode == 'train':
        model.train()
    model.image_size = args.image_size
    model.cuda()
    return model


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


def main(opt):
    name = 'features'
    checkpoint = torch.load(opt.checkpoint)
    rep_size = checkpoint['model_kwargs']['rep_size']
    vocab = checkpoint['model_kwargs']['vocab']
    num_objs = len(vocab['object_to_idx'])
    model = build_model(opt, checkpoint)
    train, val, test = VG.splits(transform=transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]))
    train_loader, test_loader = VGDataLoader.splits(train, test, batch_size=opt.batch_size,
                                                    num_workers=opt.loader_num_workers,
                                                    num_gpus=1)
    loader = test_loader
    print(train.ind_to_classes)

    save_path = os.path.dirname(opt.checkpoint)

    ########### Encode features ###########
    counter = 0
    max_counter = 1000000000
    print('begin')
    with torch.no_grad():
        features = {}
        for label in range(num_objs):
            features[label] = np.zeros((0, rep_size))
        for i, data in enumerate(loader):
            if counter >= max_counter:
                break
            # (all_imgs, all_objs, all_boxes, all_masks, all_triples,
            #            all_obj_to_img, all_triple_to_img, all_attributes)
            imgs = data[0].cuda()
            objs = data[1]
            objs = [j.item() for j in objs]
            boxes = data[2].cuda()
            obj_to_img = data[5].cuda()
            crops = crop_bbox_batch(imgs, boxes, obj_to_img, opt.object_size)
            feat = model.repr_net(model.image_encoder(crops)).cpu()
            for ind, label in enumerate(objs):
                features[label] = np.append(features[label], feat[ind].view(1, -1), axis=0)
            counter += len(objs)

            # print('%d / %d images' % (i + 1, dataset_size))
        save_name = os.path.join(save_path, name + '.npy')
        np.save(save_name, features)

    ############## Clustering ###########
    print('begin clustering')
    load_name = os.path.join(save_path, name + '.npy')
    features = np.load(load_name).item()
    cluster(features, num_objs, 100, save_path)
    cluster(features, num_objs, 10, save_path)
    cluster(features, num_objs, 1, save_path)


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.object_size = 64
    main(opt)
