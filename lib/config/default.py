# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PERCENT = 1.0
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

_C.CONTRASTIVE = CN()
_C.CONTRASTIVE.IS_CONTRASTIVE = True
_C.CONTRASTIVE.EMB_SIZE = 64
_C.CONTRASTIVE.REPR_SIZE = [1024]
_C.CONTRASTIVE.NUM_SAMPLES = 2
_C.CONTRASTIVE.EVAL_HEAD = False
_C.CONTRASTIVE.TAU = 0.1
_C.CONTRASTIVE.NORMALIZE = True
_C.CONTRASTIVE.HEAD = CN()
_C.CONTRASTIVE.HEAD.LAYERS = 2
_C.CONTRASTIVE.HEAD.SIZE = 512
_C.CONTRASTIVE.HEAD.ADD_BN = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.FROZEN = False
_C.MODEL.DOWNSAMPLE = False
_C.MODEL.NAME = 'mdeq'       # Default for classification
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.WNORM = False
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.NUM_LAYERS = 5
_C.MODEL.NUM_GROUPS = 4
_C.MODEL.DROPOUT = 0.0
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.F_THRES = 30
_C.MODEL.B_THRES = 40
_C.MODEL.DOWNSAMPLE_TIMES = 2
_C.MODEL.EXPANSION_FACTOR = 5
_C.MODEL.TAG_PER_JOINT = True
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN()
_C.LOSS.USE_OHEM = False
_C.LOSS.OHEMTHRES = 0.9
_C.LOSS.OHEMKEEP = 100000
_C.LOSS.CLASS_BALANCE = True
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.NUM_CLASSES = 1000
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.EXTRA_TRAIN_SET = ''


# training data augmentation
_C.DATASET.AUGMENT = True
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.0
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = False

_C.DATASET.AUGMENTATIONS = CN()
_C.DATASET.AUGMENTATIONS.cj0 = 0.4
_C.DATASET.AUGMENTATIONS.cj1 = 0.4
_C.DATASET.AUGMENTATIONS.cj2 = 0.4
_C.DATASET.AUGMENTATIONS.cj3 = 0.1
_C.DATASET.AUGMENTATIONS.cj_p = 0.1
_C.DATASET.AUGMENTATIONS.gs_p = 0.8
_C.DATASET.AUGMENTATIONS.crop_s0 = 0.2
_C.DATASET.AUGMENTATIONS.crop_s1 = 1.0
_C.DATASET.AUGMENTATIONS.crop_r0 = 0.75
_C.DATASET.AUGMENTATIONS.crop_r1 = 1.33
_C.DATASET.AUGMENTATIONS.hf_p = 0.5

_C.CLF = CN()
_C.CLF.BATCH_SIZE_PER_GPU = 32

# train
_C.TRAIN = CN()
_C.TRAIN.IMAGE_SIZE = [1024, 512]  # width * height
_C.TRAIN.BASE_SIZE = 2048
_C.TRAIN.DOWNSAMPLERATE = 1
_C.TRAIN.FLIP = True
_C.TRAIN.MULTI_SCALE = True
_C.TRAIN.SCALE_FACTOR = 16
_C.TRAIN.CLIP = -1.0

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_SCHEDULER = 'step'
_C.TRAIN.EXTRA_LR = 0.001

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.PRETRAIN_STEPS = 100000
_C.TRAIN.IGNORE_LABEL = -1

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.EXTRA_EPOCH = 0

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.MODEL_FILE = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32

# For segmentation
_C.TEST.IMAGE_SIZE = [2048, 1024]  # width * height
_C.TEST.BASE_SIZE = 2048
_C.TEST.NUM_SAMPLES = 0

# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.MULTI_SCALE = False
_C.TEST.CENTER_CROP_TEST = False
_C.TEST.SCALE_LIST = [1]
_C.TEST.USE_GT_BBOX = False

# nms
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.NMS_THRE = 0.6
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.5
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    if args.dataDir:
        cfg.DATA_DIR = args.dataDir

    if args.testModel:
        cfg.TEST.MODEL_FILE = args.testModel

    if args.percent < 1:
        cfg.PERCENT = args.percent


    extra = []
    if args.frozen:
        cfg.MODEL.FROZEN = args.frozen
        extra.append('frozen')

    if args.pretrained:
        cfg.TEST.MODEL_FILE = args.pretrained
        cfg.MODEL.PRETRAINED = args.pretrained
        extra.append('pretrained')

    if args.downsample:
        cfg.MODEL.DOWNSAMPLE = args.downsample
        extra.append('downsample')

    if extra:
        extra = '_'.join(extra)
        cfg.OUTPUT_DIR += extra

    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

