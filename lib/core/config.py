# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

from lib.utils.string_parser import parse_string_to_keyvalue

config = edict()

config.TRANSFORMER = 'multi_view_pose_transformer'

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'
config.MODEL = 'multi_view_pose_transformer'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100

# higherhrnet definition
config.MODEL_EXTRA = edict()
config.MODEL_EXTRA.PRETRAINED_LAYERS = ['*']
config.MODEL_EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL_EXTRA.STEM_INPLANES = 64

config.MODEL_EXTRA.STAGE2 = edict()
config.MODEL_EXTRA.STAGE2.NUM_MODULES = 1
config.MODEL_EXTRA.STAGE2.NUM_BRANCHES = 2
config.MODEL_EXTRA.STAGE2.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
config.MODEL_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
config.MODEL_EXTRA.STAGE2.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE3 = edict()
config.MODEL_EXTRA.STAGE3.NUM_MODULES = 4
config.MODEL_EXTRA.STAGE3.NUM_BRANCHES = 3
config.MODEL_EXTRA.STAGE3.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.MODEL_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.MODEL_EXTRA.STAGE3.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE4 = edict()
config.MODEL_EXTRA.STAGE4.NUM_MODULES = 3
config.MODEL_EXTRA.STAGE4.NUM_BRANCHES = 4
config.MODEL_EXTRA.STAGE4.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.MODEL_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.MODEL_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.DECONV = edict()
config.MODEL_EXTRA.DECONV.NUM_DECONVS = 1
config.MODEL_EXTRA.DECONV.NUM_CHANNELS = 32
config.MODEL_EXTRA.DECONV.KERNEL_SIZE = 4
config.MODEL_EXTRA.DECONV.NUM_BASIC_BLOCKS = 4
config.MODEL_EXTRA.DECONV.CAT_OUTPUT = True

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.PRETRAINED_BACKBONE = ''
config.NETWORK.NUM_JOINTS = 20
config.NETWORK.INPUT_SIZE = 512
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.SIGMA = 2
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.AGGRE = True
config.NETWORK.USE_GT = False
config.NETWORK.BETA = 100.0

# pose_resnet related params
config.POSE_RESNET = edict()
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = '../data/panoptic/'
config.DATASET.TRAIN_DATASET = 'panoptic'
config.DATASET.TEST_DATASET = 'panoptic'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 2
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True
config.DATASET.COLOR_RGB = False
config.DATASET.FLIP = True
config.DATASET.DATA_AUGMENTATION = True
config.DATASET.CAMERA_NUM = 5
config.DATASET.DATA_ROOT = None
config.DATASET.MAX_DATA_NUM = None
config.DATASET.ADD_VOXEL_PRED = None
config.DATASET.TRAIN_CAM_SEQ = 'CMU0_ori'
config.DATASET.TEST_CAM_SEQ = 'CMU0_ori'
config.DATASET.CAMERA_DETAIL = False
config.DATASET.NMS_DETAIL = False
config.DATASET.NMS_DETAIL_ALL = False

# Dataset selection for debugging
config.DATASET.SUBSET_SELECTION = 'all'
config.DATASET.FILTER_VALID_OBSERVATIONS = False

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 0
config.DATASET.PESUDO_GT = None

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [20]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.FINETUNE_MODEL = None

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True
config.TRAIN.clip_max_norm = 0.1


# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = 'best'
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.HEATMAP_LOCATION_FILE = 'predicted_heatmaps.h5'
config.TEST.PRED_FILE = None

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True
config.DEBUG.PRINT_TO_FILE = False

config.DEBUG.LOG_VAL_LOSS = True

config.DEBUG.VISUALIZATION_JUMP_NUM = -1  # open visualization if this >= 0
config.DEBUG.WANDB_KEY = ''
config.DEBUG.WANDB_NAME = ''

# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.PAIRWISE_FILE = ''
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = np.array([2000.0, 2000.0, 2000.0])
config.PICT_STRUCT.CUBE_SIZE = np.array([64, 64, 64])
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False

config.MULTI_PERSON = edict()
config.MULTI_PERSON.SPACE_SIZE = np.array([4000.0, 5200.0, 2400.0])
config.MULTI_PERSON.SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.MULTI_PERSON.INITIAL_CUBE_SIZE = np.array([24, 32, 16])
config.MULTI_PERSON.MAX_PEOPLE_NUM = 10
config.MULTI_PERSON.THRESHOLD = 0.1

config.DECODER = dict()
config.DECODER.d_model = 256
config.DECODER.nhead = 8
config.DECODER.dim_feedforward = 1024
config.DECODER.dropout = 0.1
config.DECODER.activation = 'relu'
config.DECODER.num_feature_levels = 1
config.DECODER.dec_n_points = 4
config.DECODER.num_decoder_layers = 6
config.DECODER.return_intermediate_dec = True
config.DECODER.num_instance = 10
config.DECODER.num_keypoints = 15
config.DECODER.num_views = 5
config.DECODER.with_pose_refine = True
config.DECODER.aux_loss = True
config.DECODER.lr_linear_proj_mult = 0.1
config.DECODER.loss_pose_normalize = False
config.DECODER.loss_joint_type = 'l1'
config.DECODER.pred_class_fuse = 'mean'
config.DECODER.pred_conf_threshold = 0.5

# YOLO backbone configuration
config.YOLO = edict()
config.YOLO.ENABLED = False
config.YOLO.MODEL_VARIANT = 'v9-c'  # Options: v9-n, v9-t, v9-s, v9-c, v9-e, v9-m
config.YOLO.PRETRAINED_WEIGHTS = ''
config.YOLO.FREEZE_BACKBONE = True
config.YOLO.FEATURE_LEVELS = [0, 1, 2]
config.YOLO.USE_DETECTION_GUIDANCE = True
config.YOLO.NUM_WHOLEBODY_CLASSES = 34
config.DECODER.match_coord_est = 'abs'
config.DECODER.match_coord_gt = 'norm'
config.DECODER.detach_refpoints_cameraprj_firstlayer = True
config.DECODER.fuse_view_feats = 'cat_proj'
config.DECODER.use_loss_pose_perbone = False
config.DECODER.use_loss_pose_perjoint_aligned = False
config.DECODER.use_loss_pose_perprojection = False
config.DECODER.use_loss_pose_perprojection_2d = True
config.DECODER.use_quality_focal_loss = False

config.DECODER.loss_weight_loss_ce = 2.
config.DECODER.loss_pose_perjoint = 5.
config.DECODER.loss_pose_perbone = 5.
config.DECODER.loss_pose_perjoint_aligned = 5.
config.DECODER.loss_heatmap2d = 2.
config.DECODER.loss_pose_perprojection_2d = 5.

config.DECODER.epipolar_encoder = False

config.DECODER.REGRESS_GRID_SIZE = [200.0, 200.0, 200.0]
config.DECODER.REGRESS_CUBE_SIZE = [20, 20, 20]
config.DECODER.agnostic_v2vnet = False

config.DECODER.voxel_regression_type = 'perjoint'
config.DECODER.pose_embed_layer = 3

config.DECODER.query_embed_type = 'person_joint'
config.DECODER.optimizer = 'adam'
config.DECODER.lr_decay_epoch = [40, ]

config.DECODER.projattn_posembed_mode = 'no_use'

config.DECODER.use_feat_level = [0, 1, 2]
config.DECODER.query_adaptation = True
config.DECODER.inference_conf_thr = [0.5, ]
# used for finetuning trained mvp on one dataset
# to another dataset (i.e., here panoptic to shelf/campus)
config.DECODER.convert_joint_format_indices = None

config.DECODER.t_pose_dir = './t_pose.pt'
config.DECODER.feature_update_method = 'MLP'
config.DECODER.init_self_attention = False
config.DECODER.open_forward_ffn = False
config.DECODER.query_filter_method = 'threshold'

config.DECODER.init_ref_method = 'sample_space'
config.DECODER.init_ref_method_value = None

config.DECODER.gt_match = True
config.DECODER.close_pose_embedding = False

config.DECODER.share_layer_weights = False

config.DECODER.bayesian_update = False
config.DECODER.triangulation_method = 'linalg'

config.DECODER.decay_method = 'none'

config.DECODER.gt_match_test = False

config.DECODER.match_method = 'hungarian'   # hungarian, multiple
config.DECODER.match_method_value = 300

config.DECODER.use_ce_match = False

config.DECODER.filter_query = True

# config.DECODER.open_init_loss = False
config.DECODER.loss_weight_init = 0

config.SMPL = dict()
config.SMPL.pred_smpl = False
config.SMPL.pred_smpl_fuse = 'mean_before_pred'
config.SMPL.concat_smpl = True
config.SMPL.smpl_embed_layer = 3
config.SMPL.init_param_file = 'data/neutral_smpl_mean_params.h5'
config.SMPL.smpl_file = 'data/smpl'
config.SMPL.loss_smpl_3d = 1.0
config.SMPL.loss_smpl_2d = 1.0
config.SMPL.loss_smpl_adv = 1.0



def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

# input: unknown args
def update_config_dynamic_input(args):
    key_value_items = parse_string_to_keyvalue(args)
    for item in key_value_items:
        for k in item:
            v = item[k]
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                print("{} not exist in config.py".format(k))
                # raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR,
                                       config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR,
                                         config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=cfg.POSE_RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.POSE_RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
