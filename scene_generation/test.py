#!/usr/bin/python
import os
import json
from collections import defaultdict
import torch

from scene_generation.args import get_args
from combine_sg2im_neural_motifs.bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
from torchvision import transforms
from scene_generation.trainer import Trainer
from scene_generation.data import imagenet_deprocess_batch
from scipy.misc import imsave


def makedir(base, name, flag=True):
    dir_name = None
    if flag:
        dir_name = os.path.join(base, name)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
    return dir_name


def check_model(args, loader, model):
    num_samples = 0
    model.forward_D = False
    model.eval()

    img_dir = makedir(args.output_dir, 'test')
    gt_img_dir = makedir(args.output_dir, 'test_real')

    with torch.no_grad():
        for batch in loader:
            result = model[batch]
            imgs, imgs_pred = result.imgs, result.imgs_pred

            imgs_gt = imagenet_deprocess_batch(imgs)
            imgs_pred = imagenet_deprocess_batch(imgs_pred)
            for i in range(imgs_pred.size(0)):
                img_filename = '%04d.png' % num_samples

                img_gt = imgs_gt[i].numpy().transpose(1, 2, 0)
                img_gt_path = os.path.join(gt_img_dir, img_filename)
                imsave(img_gt_path, img_gt)

                img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                img_path = os.path.join(img_dir, img_filename)
                imsave(img_path, img_pred_np)

                num_samples += 1

            print('Saved %d images' % num_samples)
            if num_samples >= args.num_val_samples:
                break


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
    check_model(args, test_loader, trainer)


if __name__ == '__main__':
    args = get_args()
    main(args)
