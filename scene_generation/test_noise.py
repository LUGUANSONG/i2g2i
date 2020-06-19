#!/usr/bin/python
import os
import json
from collections import defaultdict
import torch
from copy import deepcopy

from scene_generation.args import get_args
from combine_sg2im_neural_motifs.bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
from torchvision import transforms
from scene_generation.trainer import Trainer
from scene_generation.data import imagenet_deprocess_batch
from scipy.misc import imsave
import pickle
from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 16)


def draw_box(draw, boxx, text_str, color_style='normal'):
    box = tuple([float(b) for b in boxx])
    if color_style == "special":
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

    img_dir = makedir(args.output_dir, 'test_noise' if args.use_gt_textures else 'test_noise_patch')

    crops_path = os.path.join(args.output_dir, args.features_file_name[:-4] + "_crops.pkl")
    print(crops_path)
    if os.path.isfile(crops_path):
        crops_dict = pickle.load(open(crops_path, "rb"))
    else:
        crops_dict = None
        print('No crops file !!!!!!!!!!!!!')

    image_size = 256
    use_gt_textures = args.use_gt_textures
    args.use_gt_textures = True
    with torch.no_grad():
        for _batch in loader:
            for noise_index in range(args.num_diff_noise):
                if noise_index > 0:
                    args.use_gt_textures = use_gt_textures
                batch = deepcopy(_batch)
                result = model[batch]
                imgs, imgs_pred = result.imgs, result.imgs_pred
                objs = result.objs
                change_indexes = result.change_indexes
                crop_indexes = result.crop_indexes
                boxes = result.boxes
                obj_to_img = result.obj_to_img

                imgs_pred = imagenet_deprocess_batch(imgs_pred)
                for i in range(imgs_pred.size(0)):
                    this_img_dir = makedir(img_dir, "%d" % (num_samples + i))
                    img_filename = '%04d.png' % noise_index

                    img_pred_np = imgs_pred[i].numpy().transpose(1, 2, 0)
                    img_path = os.path.join(this_img_dir, img_filename)
                    imsave(img_path, img_pred_np)

                    # draw bbox and class
                    image = torch.ones(3, image_size, image_size)
                    image = transforms.ToPILImage()(image).convert("RGB")
                    draw = ImageDraw.Draw(image)
                    index = (obj_to_img == i).nonzero()[:, 0]
                    for ind in index:
                        box = boxes[ind]
                        cls = objs[ind]
                        color_style = 'normal'
                        draw = draw_box(draw, box * image_size, loader.dataset.ind_to_classes[cls + 1], color_style)

                    # draw box of changed object and save used object patch
                    if change_indexes is not None:
                        change_index = change_indexes[i]
                        box = boxes[change_index]
                        cls = objs[change_index]
                        print(cls, type(cls))
                        color_style = 'special'
                        draw = draw_box(draw, box * image_size, loader.dataset.ind_to_classes[cls + 1], color_style)

                        if args.save_crop and crops_dict is not None:
                            crop = crops_dict[cls][crop_indexes[i]]
                            crop = crop.numpy().transpose(1, 2, 0)
                            crop_path = os.path.join(this_img_dir, "%04d_crop.png" % noise_index)
                            imsave(crop_path, crop)

                    if args.save_layout:
                        image.save(os.path.join(this_img_dir, "%04d_layout.png" % noise_index))

            num_samples += imgs.shape[0]
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
