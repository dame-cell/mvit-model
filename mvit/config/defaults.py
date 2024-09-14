from fvcore.common.config import CfgNode

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True
_C.TRAIN.DATASET = "imagenet"
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.EVAL_PERIOD = 10
_C.TRAIN.CHECKPOINT_PERIOD = 10
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT_FILE_PATH = ""
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False
_C.TRAIN.MIXED_PRECISION = False

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()
_C.AUG.NUM_SAMPLE = 1
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AA_TYPE = "rand-m9-n6-mstd0.5-inc1"
_C.AUG.INTERPOLATION = "bicubic"
_C.AUG.RE_PROB = 0.25
_C.AUG.RE_MODE = "pixel"
_C.AUG.RE_COUNT = 1
_C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# MixUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()
_C.MIXUP.ENABLE = True
_C.MIXUP.ALPHA = 0.8
_C.MIXUP.CUTMIX_ALPHA = 1.0
_C.MIXUP.PROB = 1.0
_C.MIXUP.SWITCH_PROB = 0.5
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()
_C.TEST.ENABLE = False
_C.TEST.DATASET = "imagenet"
_C.TEST.BATCH_SIZE = 64
_C.TEST.CHECKPOINT_FILE_PATH = ""
_C.TEST.CHECKPOINT_SQUEEZE_TEMPORAL = True

# ----------------------------------------------------------------------------- #
# Model options
# ----------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.MODEL_NAME = "MViT"
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.LOSS_FUNC = "soft_cross_entropy"
_C.MODEL.DROPOUT_RATE = 0.0
_C.MODEL.HEAD_ACT = "softmax"
_C.MODEL.ACT_CHECKPOINT = False

# ----------------------------------------------------------------------------- #
# MViT options
# ----------------------------------------------------------------------------- #
_C.MVIT = CfgNode()
_C.MVIT.MODE = "conv"
_C.MVIT.POOL_FIRST = False
_C.MVIT.CLS_EMBED_ON = False
_C.MVIT.PATCH_KERNEL = [7, 7]
_C.MVIT.PATCH_STRIDE = [4, 4]
_C.MVIT.PATCH_PADDING = [3, 3]
_C.MVIT.EMBED_DIM = 96
_C.MVIT.NUM_HEADS = 1
_C.MVIT.MLP_RATIO = 4.0
_C.MVIT.QKV_BIAS = True
_C.MVIT.DROPPATH_RATE = 0.1
_C.MVIT.DEPTH = 16
_C.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
_C.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [8, 2.0]]
_C.MVIT.POOL_KV_STRIDE = [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]
_C.MVIT.POOL_Q_STRIDE: [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1], [8, 2, 2], [9, 1, 1]]
_C.MVIT.POOL_KVQ_KERNEL = [3, 3]
_C.MVIT.ZERO_DECAY_POS_CLS = False
_C.MVIT.USE_ABS_POS = False
_C.MVIT.REL_POS_SPATIAL = True
_C.MVIT.REL_POS_ZERO_INIT = False
_C.MVIT.RESIDUAL_POOLING = True
_C.MVIT.DIM_MUL_IN_ATT = True

# ----------------------------------------------------------------------------- #
# Data options
# ----------------------------------------------------------------------------- #
_C.DATA = CfgNode()
_C.DATA.PATH_TO_DATA_DIR = ""
_C.DATA.PATH_TO_PRELOAD_IMDB = ""
_C.DATA.MEAN = [0.485, 0.456, 0.406]
_C.DATA.STD = [0.229, 0.224, 0.225]
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.TEST_CROP_SIZE = 224
_C.DATA.VAL_CROP_RATIO = 0.875
_C.DATA.IN22K_TRAINVAL = False
_C.DATA.IN22k_VAL_IN1K = ""

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.BASE_LR = 0.00025
_C.SOLVER.LR_POLICY = "cosine"
_C.SOLVER.COSINE_END_LR = 1e-6
_C.SOLVER.STEP_SIZE = 1
_C.SOLVER.STEPS = []
_C.SOLVER.LRS = []
_C.SOLVER.MAX_EPOCH = 300
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.DAMPENING = 0.0
_C.SOLVER.NESTEROV = True
_C.SOLVER.WEIGHT_DECAY = 0.05
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCHS = 70.0
_C.SOLVER.WARMUP_START_LR = 1e-8
_C.SOLVER.OPTIMIZING_METHOD = "sgd"
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False
_C.SOLVER.COSINE_AFTER_WARMUP = True
_C.SOLVER.ZERO_WD_1D_PARAM = True
_C.SOLVER.CLIP_GRAD_VAL = None
_C.SOLVER.CLIP_GRAD_L2NORM = None
_C.SOLVER.LAYER_DECAY = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.NUM_GPUS = 8
_C.NUM_SHARDS = 1
_C.SHARD_ID = 0
_C.OUTPUT_DIR = "./tmp"
_C.RNG_SEED = 0
_C.LOG_PERIOD = 10
_C.LOG_MODEL_INFO = True
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True


def assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
