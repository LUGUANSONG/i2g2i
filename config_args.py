"""
Configuration file!
"""
import os
from argparse import ArgumentParser
import numpy as np

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def path(fn):
    return os.path.join(DATA_PATH, fn)

def stanford_path(fn):
    return os.path.join(DATA_PATH, 'stanford_filtered', fn)

# =============================================================================
# Update these with where your data is stored ~~~~~~~~~~~~~~~~~~~~~~~~~

VG_IMAGES = '/newNAS/Workspaces/UCGroup/gslu/aws_ailab/datasets/vg/VG_100K'
RCNN_CHECKPOINT_FN = path('faster_rcnn_500k.h5')

IM_DATA_FN = stanford_path('image_data.json')
VG_SGG_FN = stanford_path('VG-SGG.h5')
VG_SGG_DICT_FN = stanford_path('VG-SGG-dicts.json')
PROPOSAL_FN = stanford_path('proposals.h5')

COCO_PATH = '/home/rowan/datasets/mscoco'
# =============================================================================
# =============================================================================


MODES = ('sgdet', 'sgcls', 'predcls')

BOX_SCALE = 1024  # Scale at which we have the boxes
IM_SCALE = 592      # Our images will be resized to this res without padding

# Proposal assignments
BG_THRESH_HI = 0.5
BG_THRESH_LO = 0.0

RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
RPN_FG_FRACTION = 0.5
FG_FRACTION = 0.25
# Total number of examples
RPN_BATCHSIZE = 256
ROIS_PER_IMG = 256
REL_FG_FRACTION = 0.25
RELS_PER_IMG = 256

RELS_PER_IMG_REFINE = 64

BATCHNORM_MOMENTUM = 0.01
ANCHOR_SIZE = 16

ANCHOR_RATIOS = (0.23232838, 0.63365731, 1.28478321, 3.15089189) #(0.5, 1, 2)
ANCHOR_SCALES = (2.22152954, 4.12315647, 7.21692515, 12.60263013, 22.7102731) #(4, 8, 16, 32)


# ******* parser ********
parser = ArgumentParser(description='training code')

# Options to deprecate
parser.add_argument('-coco', dest='coco', help='Use COCO (default to VG)', action='store_true')
parser.add_argument('-ckpt', dest='ckpt', help='Filename to load from', type=str, default='')
parser.add_argument('-det_ckpt', dest='det_ckpt', help='Filename to load detection parameters from', type=str, default='')

parser.add_argument('-save_dir', dest='save_dir',
                    help='Directory to save things to, such as checkpoints/save', default='', type=str)

parser.add_argument('-ngpu', dest='num_gpus', help='cuantos GPUs tienes', type=int, default=3)
parser.add_argument('-nwork', dest='num_workers', help='num processes to use as workers', type=int, default=1)

parser.add_argument('-lr', dest='lr', help='learning rate', type=float, default=1e-3)

parser.add_argument('-b', dest='batch_size', help='batch size per GPU',type=int, default=2)
parser.add_argument('-val_size', dest='val_size', help='val size to use (if 0 we wont use val)', type=int, default=5000)

parser.add_argument('-l2', dest='l2', help='weight decay', type=float, default=1e-4)
parser.add_argument('-clip', dest='clip', help='gradients will be clipped to have norm less than this', type=float, default=5.0)
parser.add_argument('-p', dest='print_interval', help='print during training', type=int,
                    default=100)
parser.add_argument('-m', dest='mode', help='mode \in {sgdet, sgcls, predcls}', type=str,
                    default='sgdet')
parser.add_argument('-model', dest='model', help='which model to use? (motifnet, stanford). If you want to use the baseline (NoContext) model, then pass in motifnet here, and nl_obj, nl_edge=0', type=str,
                    default='motifnet')
parser.add_argument('-old_feats', dest='old_feats', help='Use the original image features for the edges', action='store_true')
parser.add_argument('-order', dest='order', help='Linearization order for Rois (confidence -default, size, random)',
                    type=str, default='confidence')
parser.add_argument('-cache', dest='cache', help='where should we cache predictions', type=str,
                    default='')
parser.add_argument('-gt_box', dest='gt_box', help='use gt boxes during training', action='store_true')
parser.add_argument('-adam', dest='adam', help='use adam. Not recommended', action='store_true')
parser.add_argument('-test', dest='test', help='test set', action='store_true')
parser.add_argument('-multipred', dest='multi_pred', help='Allow multiple predicates per pair of box0, box1.', action='store_true')
parser.add_argument('-nepoch', dest='num_epochs', help='Number of epochs to train the model for',type=int, default=25)
parser.add_argument('-resnet', dest='use_resnet', help='use resnet instead of VGG', action='store_true')
parser.add_argument('-proposals', dest='use_proposals', help='Use Xu et als proposals', action='store_true')
parser.add_argument('-nl_obj', dest='nl_obj', help='Num object layers', type=int, default=1)
parser.add_argument('-nl_edge', dest='nl_edge', help='Num edge layers', type=int, default=2)
parser.add_argument('-hidden_dim', dest='hidden_dim', help='Num edge layers', type=int, default=256)
parser.add_argument('-pooling_dim', dest='pooling_dim', help='Dimension of pooling', type=int, default=4096)
parser.add_argument('-pass_in_obj_feats_to_decoder', dest='pass_in_obj_feats_to_decoder', action='store_true')
parser.add_argument('-pass_in_obj_feats_to_edge', dest='pass_in_obj_feats_to_edge', action='store_true')
parser.add_argument('-rec_dropout', dest='rec_dropout', help='recurrent dropout to add', type=float, default=0.1)
parser.add_argument('-use_bias', dest='use_bias',  action='store_true')
parser.add_argument('-use_tanh', dest='use_tanh',  action='store_true')
parser.add_argument('-limit_vision', dest='limit_vision',  action='store_true')

# sg2im options
from sg2im.utils import int_tuple, float_tuple, str_tuple, bool_flag
# parser.add_argument('--dataset', default='coco', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--not_imagenet_preprocess', default=False, type=bool_flag)
parser.add_argument('--no_rescale', default=False, type=bool_flag)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=-1, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--save_gt', action='store_true')
parser.add_argument('--num_diff_noise', default=10, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# # Generator options
# parser.add_argument('--mask_size', default=16, type=int)  # Set this to 0 to use no masks
parser.add_argument('--not_decrease_feature_dimension', default=False, type=bool_flag)
parser.add_argument('--mask_size', default=0, type=int)  # Set this to 0 to use no masks
parser.add_argument('--object_no_noise_with_mask', default=True, type=bool_flag)
parser.add_argument('--object_no_noise_with_bbox', default=False, type=bool_flag)
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--layout_noise_only_on_foreground', default=False, type=bool_flag)
parser.add_argument('--object_noise_dim', default=0, type=int)
parser.add_argument('--noise_apply_method', default="concat", type=str)
parser.add_argument('--noise_mask_ratio', type=float, default=0.0)
parser.add_argument('--noise_std', type=float, default=1)
parser.add_argument('--noise_std_mode', type=str, default='fix', help='can be fix, change')
parser.add_argument('--noise_std_change_iters', type=str, default="-1")
parser.add_argument('--noise_std_change_vals', type=str, default="")
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--l1_on_bg', default=False, type=bool_flag)
parser.add_argument('--l1_mode', type=str, default='fix', help='can be fix, change')
parser.add_argument('--l1_change_iters', type=str, default="-1")
parser.add_argument('--l1_change_vals', type=str, default="")
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float)  # DEPRECATED
## bicyclegan losses
parser.add_argument('--e_img_arch',
                    default='C3-64-1,C3-32-1')
parser.add_argument('--kl_loss_weight', default=0.01, type=float)
parser.add_argument('--z_random_rec_loss_weight', default=0.5, type=float)
parser.add_argument('--z_random_rec_train_encoder', default=False, type=bool_flag)
parser.add_argument('--crops_encoded_rec_loss_weight', default=10., type=float)
parser.add_argument('--imgs_encoded_rec_loss_weight', default=10., type=float)

# Perceptual loss
parser.add_argument('--perceptual_loss_weight', default=0.0, type=float)
parser.add_argument('--perceptual_on_bg', default=False, type=bool_flag)
parser.add_argument('--perceptual_not_on_noise', default=False, type=bool_flag)

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', choices=['gan', 'lsgan', 'wgan-gp'], default='gan')
parser.add_argument('--n_critic', type=int, default=0)
parser.add_argument('--n_gen', type=int, default=1)
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
                    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight
parser.add_argument('--d_obj_mode', type=str, default='fix', help='can be fix, change, change_linear')
parser.add_argument('--d_obj_change_iters', type=str, default="-1")
parser.add_argument('--d_obj_change_vals', type=str, default="")
parser.add_argument('--d_obj_gp_weight', type=float, default=10) # multiplied by d_loss_weight
parser.add_argument('--d_obj_rec_feat_weight', default=0.0, type=float)

parser.add_argument('--ac_loss_weight', default=0.1, type=float)
parser.add_argument('--ac_loss_mode', type=str, default='fix', help='can be fix, change, change_linear')
parser.add_argument('--ac_loss_change_iters', type=str, default="-1")
parser.add_argument('--ac_loss_change_vals', type=str, default="")

# Image discriminator
parser.add_argument('--d_img_arch',
                    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float)  # multiplied by d_loss_weight
parser.add_argument('--d_img_mode', type=str, default='fix', help='can be fix, change, change_linear')
parser.add_argument('--d_img_change_iters', type=str, default="-1")
parser.add_argument('--d_img_change_vals', type=str, default="")
parser.add_argument('--d_img_gp_weight', type=float, default=10) # multiplied by d_loss_weight
parser.add_argument('--down_to_1channel', default=False, type=bool_flag)
parser.add_argument('--condition_d_img', action='store_true', help='image gan conditioned on layout?')
parser.add_argument('--condition_d_img_on_class_label_map', default=False, type=bool_flag)

# Background discriminator
parser.add_argument('--d_bg_arch',
                    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_bg_weight', default=0.0, type=float)  # multiplied by d_loss_weight
parser.add_argument('--d_bg_mode', type=str, default='fix', help='can be fix, change, change_linear')
parser.add_argument('--d_bg_change_iters', type=str, default="-1")
parser.add_argument('--d_bg_change_vals', type=str, default="")
parser.add_argument('--d_bg_gp_weight', type=float, default=10) # multiplied by d_loss_weight
parser.add_argument('--condition_d_bg', action='store_true', help='bg gan conditioned on layout?')

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)
parser.add_argument('--output_subdir', type=str, default="test_noise")
parser.add_argument('--exchange_feat_cls', default=False, type=bool_flag)
parser.add_argument('--change_bbox', default=False, type=bool_flag)

config_args = parser.parse_args()
# IM_SCALE = self.image_size[0]
# print("***************** manual set IM_SCALE to image_size: %d" % IM_SCALE)
if config_args.l1_mode in ["change", "change_linear"]:
    config_args.l1_change_iters = [int(x) for x in config_args.l1_change_iters.split(",")]
    config_args.l1_change_vals = [float(x) for x in config_args.l1_change_vals.split(",")]
if config_args.noise_std_mode in ["change", "change_linear"]:
    config_args.noise_std_change_iters = [int(x) for x in config_args.noise_std_change_iters.split(",")]
    config_args.noise_std_change_vals = [float(x) for x in config_args.noise_std_change_vals.split(",")]
if config_args.d_obj_mode in ["change", "change_linear"]:
    config_args.d_obj_change_iters = [int(x) for x in config_args.d_obj_change_iters.split(",")]
    config_args.d_obj_change_vals = [float(x) for x in config_args.d_obj_change_vals.split(",")]
if config_args.ac_loss_mode in ["change", "change_linear"]:
    config_args.ac_loss_change_iters = [int(x) for x in config_args.ac_loss_change_iters.split(",")]
    config_args.ac_loss_change_vals = [float(x) for x in config_args.ac_loss_change_vals.split(",")]
if config_args.d_img_mode in ["change", "change_linear"]:
    config_args.d_img_change_iters = [int(x) for x in config_args.d_img_change_iters.split(",")]
    config_args.d_img_change_vals = [float(x) for x in config_args.d_img_change_vals.split(",")]

if len(config_args.ckpt) != 0:
    config_args.ckpt = os.path.join(ROOT_PATH, config_args.ckpt)
else:
    config_args.ckpt = None

if len(config_args.cache) != 0:
    config_args.cache = os.path.join(ROOT_PATH, config_args.cache)
else:
    config_args.cache = None

if len(config_args.save_dir) == 0:
    config_args.save_dir = None
else:
    config_args.save_dir = os.path.join(ROOT_PATH, config_args.save_dir)
    if not os.path.exists(config_args.save_dir):
        os.makedirs(config_args.save_dir)

assert config_args.val_size >= 0

if config_args.mode not in MODES:
    raise ValueError("Invalid mode: mode must be in {}".format(MODES))

if config_args.model not in ('motifnet', 'stanford'):
    raise ValueError("Invalid model {}".format(config_args.model))

if config_args.ckpt is not None and not os.path.exists(config_args.ckpt):
    raise ValueError("Ckpt file ({}) doesnt exist".format(config_args.ckpt))

# ******* parser end ********

