import torch
import torch.nn
import torch.utils.data as data
import argparse
import os

from configs.defaults import cfg
from src.data import MRAProjected
from src.checkpoint import CheckPointer
from src.network import BtrflyNet
from src.solver import *


def train(cfg, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE
    model = BtrflyNet(cfg)

    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    optim = torch.optim.Adam
