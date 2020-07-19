from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.USE_GAN = 0
_C.MODEL.IMAGE_SIZE = 512
_C.MODEL.USE_BN = 1
_C.MODEL.CHANNELS = (4, 32, 64, 128, 256, 256, 512, 1024, 512, 512, 256, 128, 64, 25)

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 1e-3
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.SAVE_NUM = 25

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 10
_C.TEST.DEVICE = 'cuda'

_C.ORIGINAL_PATH = "/home/dhaval/adam_data"
_C.MIP_PATH = "/home/dhaval/mip"

_C.TRAIN_SPLIT_FILE = "/home/dhaval/mip/train.txt"
_C.VAL_SPLIT_FILE = "/home/dhaval/mip/val.txt"

_C.OUTPUT_PATH = "/home/dhaval/outputs"

cfg = _C