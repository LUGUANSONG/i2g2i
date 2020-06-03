#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import exists, join
import math
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

# sg2im
from sg2im.losses import get_gan_losses, VGGLoss, gradient_penalty
from sg2im.utils import timeit, bool_flag, LossManager

# neural motifs
# from dataloaders.visual_genome import VGDataLoader, VG
# from dataloaders.mscoco import CocoDetection, CocoDataLoader
from torchvision import transforms
from bbox_feature_dataset.bbox_feature_dataset import VG, VGDataLoader
# from config import ModelConfig
from config_args import config_args

# combine
from model_bbox_feature import build_obj_discriminator, build_img_discriminator

torch.backends.cudnn.benchmark = True

def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

args = config_args

print(args)
if not exists(args.output_dir):
    os.makedirs(args.output_dir)
summary_writer = SummaryWriter(args.output_dir)

train_dataset = CIFAR10("/home/ubuntu/scene_graph/datasets/CIFAR10", train=True, transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ]), download=True)
val_dataset = CIFAR10("/home/ubuntu/scene_graph/datasets/CIFAR10", train=False, transform=transforms.Compose([
                                    transforms.Resize(args.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ]), download=True)
loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.loader_num_workers,
        'shuffle': True
    }
train_loader = DataLoader(train_dataset, **loader_kwargs)
loader_kwargs['shuffle'] = False
val_loader = DataLoader(val_dataset, **loader_kwargs)

num_class = 10
vocab = {
    'object_idx_to_name': [None] * (num_class + 1)
}

obj_discriminator, _ = build_obj_discriminator(args, vocab)
if obj_discriminator is not None:
    print(obj_discriminator)
    obj_discriminator.train()
    optimizer_d_obj = torch.optim.Adam(obj_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

img_discriminator, _ = build_img_discriminator(args)
if img_discriminator is not None:
    print(img_discriminator)
    img_discriminator.train()
    optimizer_d_img = torch.optim.Adam(img_discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))


generator = nn.ModuleList([
    nn.Linear(100, 256),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(True),
    nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(True),
    nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1),
    nn.BatchNorm2d(3),
    nn.ReLU(True),
    nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
]).cuda()
generator.train()
optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

print(generator)
gan_g_loss, gan_d_loss = get_gan_losses(args.gan_loss_type)

t = 0
epoch = 0
while True:
    if t >= args.num_iterations * (args.n_critic + 1):
        break
    epoch += 1
    print('Starting epoch %d' % epoch)

    for step, batch in enumerate(tqdm(train_loader, desc='Training Epoch %d' % epoch, total=len(train_loader))):
        t += 1

        zs = torch.randn(args.batch_size, 100).cuda()
        imgs, objs = batch
        imgs, objs = imgs.cuda(), objs.cuda()
        zs[:, 50:] = objs.view(-1, 1).repeat(1, 50)
        boxes = torch.Tensor([0, 0, 1, 1]).view(1, -1).repeat(args.batch_size, 1).cuda()
        obj_to_img = torch.arange(args.batch_size).cuda()

        imgs_pred = generator[0](zs).view(args.batch_size, 64, 2, 2)
        for i in range(1, len(generator)):
            imgs_pred = generator[i](imgs_pred)

        if t % (args.n_critic + 1) != 0:
            if obj_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_obj forward for d', args.timing):
                    d_scores_fake_crop, d_obj_scores_fake_crop, fake_crops, d_rec_feature_fake_crop = \
                        obj_discriminator(imgs_fake, objs, boxes, obj_to_img)
                    d_scores_real_crop, d_obj_scores_real_crop, real_crops, d_rec_feature_real_crop = \
                        obj_discriminator(imgs, objs, boxes, obj_to_img)
                    if args.gan_loss_type == "wgan-gp":
                        d_obj_gp = gradient_penalty(real_crops.detach(), fake_crops.detach(),
                                                    obj_discriminator.discriminator)

                ## train d
                with timeit('d_obj loss', args.timing):
                    d_obj_losses = LossManager()
                    if args.d_obj_weight > 0:
                        d_obj_gan_loss = gan_d_loss(d_obj_scores_real_crop, d_obj_scores_fake_crop)
                        d_obj_losses.add_loss(d_obj_gan_loss, 'd_obj_gan_loss')
                        if args.gan_loss_type == 'wgan-gp':
                            d_obj_losses.add_loss(d_obj_gp.mean(), 'd_obj_gp', args.d_obj_gp_weight)
                    if args.ac_loss_weight > 0:
                        d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_real_crop, objs), 'd_ac_loss_real')
                        d_obj_losses.add_loss(F.cross_entropy(d_obj_scores_fake_crop, objs), 'd_ac_loss_fake')

                with timeit('d_obj backward', args.timing):
                    optimizer_d_obj.zero_grad()
                    d_obj_losses.total_loss.backward()
                    optimizer_d_obj.step()

            if img_discriminator is not None:
                imgs_fake = imgs_pred.detach()
                with timeit('d_img forward for d', args.timing):
                    if args.condition_d_img:
                        d_scores_fake_img = img_discriminator(imgs_fake, layout)
                        d_scores_real_img = img_discriminator(imgs, layout)
                    else:
                        d_scores_fake_img = img_discriminator(imgs_fake)
                        d_scores_real_img = img_discriminator(imgs)

                    if args.gan_loss_type == "wgan-gp":
                        if args.condition_d_img:
                            d_img_gp = gradient_penalty(torch.cat([imgs, layout], dim=1), torch.cat([imgs_fake, layout], dim=1), img_discriminator)
                        else:
                            d_img_gp = gradient_penalty(imgs, imgs_fake, img_discriminator)

                with timeit('d_img loss', args.timing):
                    d_img_losses = LossManager()
                    d_img_gan_loss = gan_d_loss(d_scores_real_img, d_scores_fake_img)
                    d_img_losses.add_loss(d_img_gan_loss, 'd_img_gan_loss')
                    if args.gan_loss_type == 'wgan-gp':
                        d_img_losses.add_loss(d_img_gp.mean(), 'd_img_gp', args.d_img_gp_weight)

                with timeit('d_img backward', args.timing):
                    optimizer_d_img.zero_grad()
                    d_img_losses.total_loss.backward()
                    all_in_one_model.optimizer_d_img.step()

        if t % (args.n_critic + 1) == 0:
            ## train g
            total_loss = torch.zeros(1).to(imgs)
            losses = {}

            if obj_discriminator is not None:
                with timeit('d_obj forward for g', args.timing):
                    g_scores_fake_crop, g_obj_scores_fake_crop, _, g_rec_feature_fake_crop = \
                        obj_discriminator(imgs_pred, objs, boxes, obj_to_img)

                total_loss = add_loss(total_loss, F.cross_entropy(g_obj_scores_fake_crop, objs), losses, 'ac_loss',
                                      args.ac_loss_weight)
                weight = args.discriminator_loss_weight * args.d_obj_weight
                total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_crop), losses,
                                      'g_gan_obj_loss', weight)

            if img_discriminator is not None:
                with timeit('d_img forward for g', args.timing):
                    if args.condition_d_img:
                        g_scores_fake_img = img_discriminator(imgs_pred, layout)
                    else:
                        g_scores_fake_img = img_discriminator(imgs_pred)

                weight = args.discriminator_loss_weight * args.d_img_weight
                total_loss = add_loss(total_loss, gan_g_loss(g_scores_fake_img), losses,
                                      'g_gan_img_loss', weight)

            losses['total_loss'] = total_loss.item()

            with timeit('backward', args.timing):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        if t % (args.print_every * (args.n_critic + 1)) == 0:
            print('t = %d / %d' % (t, args.num_iterations))
            G_loss_list = []
            for name, val in losses.items():
                G_loss_list.append('[%s]: %.4f' % (name, val))
                summary_writer.add_scalar("G_%s" % name, val, t)
            print("G: %s" % ", ".join(G_loss_list))

            D_obj_loss_list = []
            for name, val in d_obj_losses.items():
                D_obj_loss_list.append('[%s]: %.4f' % (name, val))
                summary_writer.add_scalar("D_obj_%s" % name, val, t)
            print("D_obj: %s" % ", ".join(D_obj_loss_list))

            samples = {}
            samples['gt_img'] = imgs
            samples['pred_img'] = imgs_pred
            real_crops, fake_crops = real_crops, fake_crops
            samples['real_crops'] = real_crops
            samples['fake_crops'] = fake_crops

            for k, images in samples.items():
                images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
                images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
                images_min = images.min(3)[0].min(2)[0].min(1)[0].reshape(len(images), 1, 1, 1)
                images_max = images.max(3)[0].max(2)[0].max(1)[0].reshape(len(images), 1, 1, 1)
                images = images - images_min
                images = images / (images_max - images_min)
                images = images.clamp(min=0, max=1)
                samples[k] = images

            for name, images in samples.items():
                summary_writer.add_image("train_%s" % name, images, t)
