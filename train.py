import torch
import torch.nn
from torch.utils.data import DataLoader
import os

from config.defaults import cfg
from src.network import BtrflyNet
from src.solver import SolverBtrfly
from src.data import MRAProjected


def train(cfg, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE

    model = BtrflyNet(cfg)
    model = model.cuda()

    train_data = MRAProjected(cfg, mode="train")
    val_data = MRAProjected(cfg, mode="val")
    train_loader = DataLoader(train_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=4)

    lr = cfg.SOLVER.LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    optim = torch.optim.Adam

    optim_args = {"lr": lr, "weight_decay": wd}

    solver = SolverBtrfly(optim=optim, optim_args=optim_args)
    solver.train(model, train_loader, val_loader)

    # torch.save


if __name__ == "__main__":
    train(cfg)
