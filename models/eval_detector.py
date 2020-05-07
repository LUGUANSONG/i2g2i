"""
Training script 4 Detection
"""
from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.visual_genome import VGDataLoader, VG
from lib.object_detector import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
from PIL import Image, ImageDraw, ImageFont

cudnn.benchmark = True
conf = ModelConfig()

tform = [
    Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ToPILImage()
]
# tform = [
#             SquarePad(),
#             Resize(IM_SCALE),
#             ToTensor(),
#             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
transform_pipeline = Compose(tform)
font = ImageFont.truetype('/newNAS/Workspaces/UCGroup/gslu/aws_ailab/aws_server/freefont/FreeMonoBold.ttf', 12)


def draw_box(draw, boxx, text_str):
    box = tuple([float(b) for b in boxx])
    # if '-GT' in text_str:
    #     color = (255, 128, 0, 255)
    # else:
    #     color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])
    color = (255, 0, 0)

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=1)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=1)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=1)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=1)

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


if conf.coco:
    train, val = CocoDetection.splits()
    val.ids = val.ids[:conf.val_size]
    train.ids = train.ids
    train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     num_gpus=conf.num_gpus)
else:
    train, val, _ = VG.splits(num_val_im=conf.val_size, filter_non_overlap=False,
                              filter_empty_rels=False, use_proposals=conf.use_proposals)
    train_loader, val_loader = VGDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                          mode='refinerels' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet)
detector.cuda()

# Note: if you're doing the stanford setup, you'll need to change this to freeze the lower layers
if conf.use_proposals:
    for n, param in detector.named_parameters():
        if n.startswith('features'):
            param.requires_grad = False

# optimizer = optim.SGD([p for p in detector.parameters() if p.requires_grad],
#                       weight_decay=conf.l2, lr=conf.lr * conf.num_gpus * conf.batch_size, momentum=0.9)
# scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
#                               verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)

start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)
    if optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = ckpt['epoch']


def train_epoch(epoch_num):
    # detector.train()
    detector.eval()
    tr = []
    start = time.time()
    for b, batch in enumerate(val_loader):
        tr.append(train_batch(batch))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)

    :return:
    """
    '''
    return Result(
            od_obj_dists=od_obj_dists, # 1
            rm_obj_dists=obj_dists, # 2
            obj_scores=nms_scores, # 3
            obj_preds=nms_preds, # 4
            obj_fmap=obj_fmap, # 5 pick
            od_box_deltas=od_box_deltas, # 6
            rm_box_deltas=box_deltas, # 7
            od_box_targets=bbox_targets, # 8
            rm_box_targets=bbox_targets, # 9
            od_box_priors=od_box_priors, # 10
            rm_box_priors=box_priors, # 11 pick
            boxes_assigned=nms_boxes_assign, # 12
            boxes_all=nms_boxes, # 13
            od_obj_labels=obj_labels, # 14
            rm_obj_labels=rm_obj_labels, # 15
            rpn_scores=rpn_scores, # 16
            rpn_box_deltas=rpn_box_deltas, # 17
            rel_labels=rel_labels, # 18
            im_inds=im_inds, # 19 pick
            fmap=fmap if return_fmap else None, # 20
        )
    '''
    # b.imgs = F.upsample(b.imgs, size=592, mode='bilinear')
    # b.im_sizes[0, :, :2] = 592
    result = detector[b]
    print("imgs.shape", b.imgs.shape)
    print("im_sizes", b.im_sizes)
    print("boxes", result.rm_box_priors)
    print("im_inds", result.im_inds)
    print("rm_obj_dists.shape", result.rm_obj_dists.shape)

    # tform = [
    #     Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    #     Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    #     ToPILImage()
    # ]
    for i in range(len(b.imgs)):
        # pil_img = transform_pipeline(b.imgs[i]).convert("RGB")
        img_tensor = b.imgs[i].data.cpu()
        print(img_tensor.shape, img_tensor.max(), img_tensor.min())
        img_tensor = Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])(img_tensor)
        img_tensor = Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])(img_tensor)
        pil_img = ToPILImage()(img_tensor)
        pil_img = pil_img.convert("RGB")
        draw = ImageDraw.Draw(pil_img)
        for j in range(len(result.rm_box_priors)):
            if result.im_inds.data[j] == i:
                # class_ind = int(result.rm_obj_dists.data[j].max(0)[1])
                class_ind = int(result.obj_preds[j])
                class_score = float(result.obj_scores[j])
                # if class_ind != 0:
                draw = draw_box(draw, result.rm_box_priors.data[j], "%s[%.3f]" % (train.ind_to_classes[class_ind], class_score))
        pil_img.save("/newNAS/Workspaces/UCGroup/gslu/aws_ailab/code/neural-motifs/checkpoints/%d.png" % i)

    # scores = result.od_obj_dists
    # box_deltas = result.od_box_deltas
    # labels = result.od_obj_labels
    # roi_boxes = result.od_box_priors
    # bbox_targets = result.od_box_targets
    # rpn_scores = result.rpn_scores
    # rpn_box_deltas = result.rpn_box_deltas
    #
    # # detector loss
    # valid_inds = (labels.data != 0).nonzero().squeeze(1)
    # fg_cnt = valid_inds.size(0)
    # bg_cnt = labels.size(0) - fg_cnt
    # class_loss = F.cross_entropy(scores, labels)
    #
    # # No gather_nd in pytorch so instead convert first 2 dims of tensor to 1d
    # box_reg_mult = 2 * (1. / FG_FRACTION) * fg_cnt / (fg_cnt + bg_cnt + 1e-4)
    # twod_inds = valid_inds * box_deltas.size(1) + labels[valid_inds].data
    #
    # box_loss = bbox_loss(roi_boxes[valid_inds], box_deltas.view(-1, 4)[twod_inds],
    #                      bbox_targets[valid_inds]) * box_reg_mult
    #
    # loss = class_loss + box_loss
    #
    # # RPN loss
    # if not conf.use_proposals:
    #     train_anchor_labels = b.train_anchor_labels[:, -1]
    #     train_anchors = b.train_anchors[:, :4]
    #     train_anchor_targets = b.train_anchors[:, 4:]
    #
    #     train_valid_inds = (train_anchor_labels.data == 1).nonzero().squeeze(1)
    #     rpn_class_loss = F.cross_entropy(rpn_scores, train_anchor_labels)
    #
    #     # print("{} fg {} bg, ratio of {:.3f} vs {:.3f}. RPN {}fg {}bg ratio of {:.3f} vs {:.3f}".format(
    #     #     fg_cnt, bg_cnt, fg_cnt / (fg_cnt + bg_cnt + 1e-4), FG_FRACTION,
    #     #     train_valid_inds.size(0), train_anchor_labels.size(0)-train_valid_inds.size(0),
    #     #     train_valid_inds.size(0) / (train_anchor_labels.size(0) + 1e-4), RPN_FG_FRACTION), flush=True)
    #     rpn_box_mult = 2 * (1. / RPN_FG_FRACTION) * train_valid_inds.size(0) / (train_anchor_labels.size(0) + 1e-4)
    #     rpn_box_loss = bbox_loss(train_anchors[train_valid_inds],
    #                              rpn_box_deltas[train_valid_inds],
    #                              train_anchor_targets[train_valid_inds]) * rpn_box_mult
    #
    #     loss += rpn_class_loss + rpn_box_loss
    #     res = pd.Series([rpn_class_loss.data[0], rpn_box_loss.data[0],
    #                      class_loss.data[0], box_loss.data[0], loss.data[0]],
    #                     ['rpn_class_loss', 'rpn_box_loss', 'class_loss', 'box_loss', 'total'])
    # else:
    #     res = pd.Series([class_loss.data[0], box_loss.data[0], loss.data[0]],
    #                     ['class_loss', 'box_loss', 'total'])
    #
    # optimizer.zero_grad()
    # loss.backward()
    # clip_grad_norm(
    #     [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
    #     max_norm=conf.clip, clip=True)
    # optimizer.step()

    return res


def val_epoch():
    detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    for val_b, batch in enumerate(val_loader):
        vr.append(val_batch(val_b, batch))
    vr = np.concatenate(vr, 0)
    if vr.shape[0] == 0:
        print("No detections anywhere")
        return 0.0

    val_coco = val.coco
    coco_dt = val_coco.loadRes(vr)
    coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = val.ids if conf.coco else [x for x in range(len(val))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]
    return mAp


def val_batch(batch_num, b):
    result = detector[b]
    if result is None:
        return np.zeros((0, 7))
    scores_np = result.obj_scores.data.cpu().numpy()
    cls_preds_np = result.obj_preds.data.cpu().numpy()
    boxes_np = result.boxes_assigned.data.cpu().numpy()
    im_inds_np = result.im_inds.data.cpu().numpy()
    im_scales = b.im_sizes.reshape((-1, 3))[:, 2]
    if conf.coco:
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        cls_preds_np[:] = [val.ind_to_id[c_ind] for c_ind in cls_preds_np]
        im_inds_np[:] = [val.ids[im_ind + batch_num * conf.batch_size * conf.num_gpus]
                         for im_ind in im_inds_np]
    else:
        boxes_np *= BOX_SCALE / IM_SCALE
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        im_inds_np += batch_num * conf.batch_size * conf.num_gpus

    return np.column_stack((im_inds_np, boxes_np, scores_np, cls_preds_np))


print("Training starts now!")
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    mAp = val_epoch()
    scheduler.step(mAp)

    torch.save({
        'epoch': epoch,
        'state_dict': detector.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(conf.save_dir, '{}-{}.tar'.format('coco' if conf.coco else 'vg', epoch)))
