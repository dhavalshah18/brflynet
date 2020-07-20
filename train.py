import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pathlib
import os
import argparse

from config.defaults import cfg
from src.network import BtrflyNet
from src.solver import SolverBtrfly
from src.data import MRAProjected
from src.logger import *


def train(cfg, args=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICES

    model = nn.DataParallel(BtrflyNet(cfg))
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
    solver.train(model, train_loader, val_loader, num_epochs=10, log_nth=50)

    # torch.save

def main():
    parser = argparse.ArgumentParser(description="BtrflyNet training with PyTorch")
    parser.add_argument(
        "--config_file",
        default="config/btrfly.yaml",
        metavar="FILE",
        help="path to config file",
        type=str
    )
    parser.add_argument("--log_step", default=1, type=int, help="print logs every log_step")
    parser.add_argument("--save_step", default=5, type=int, help="save checkpoint every save_step")
    parser.add_argument("--eval_step", default=5, type=int, help="evaluate dataset every eval_step, disabled if eval_step <= 0")
    parser.add_argument("--use_tensorboard", default=1, type=int, help="use visdom to illustrate training process, unless use_visdom == 0")
    parser.add_argument("--train_from_no_checkpoint", default=1, type=int, help="train_from_no_checkpoint")
    args = parser.parse_args()
    
#     if torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True
        
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # make output directory designated by OUTPUT_DIR if necessary
    if cfg.OUTPUT_PATH:
        pathlib.Path(cfg.OUTPUT_PATH).mkdir(exist_ok=True)
        
    # logger_message help print message
    # it will also print info to stdout and to OUTPUT_DIR/log.txt (way: append)
    logger_message = setup_colorful_logger(
        "main_message",
        save_dir=os.path.join(cfg.OUTPUT_PATH, 'log.txt'),
        format="only_message")

    # print config info (cfg and args)
    # args are obtained by command line
    # cfg is obtained by yaml file and defaults.py in configs/
    separator(logger_message)
    logger_message.warning(" ---------------------------------------")
    logger_message.warning("|              Your config:             |")
    logger_message.warning(" ---------------------------------------")
    logger_message.info(args)
    logger_message.warning(" ---------------------------------------")
    logger_message.warning("|      Running with entire config:      |")
    logger_message.warning(" ---------------------------------------")
    logger_message.info(cfg)
    separator(logger_message)
    
    train(cfg, args)

    
if __name__ == "__main__":
    main()
