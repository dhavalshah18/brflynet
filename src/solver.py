"""Class to do training of the network."""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import pathlib

from .misc import *


class SolverBtrfly(object):
    default_optim_args = {"lr": 0.01,
                          "weight_decay": 0.}

    def __init__(self, cfg, optim=torch.optim.Adam, optim_args={},
                 loss_func=dice_loss_2arm):

        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.best_train_model = None
        self.best_val_model = None

        self._reset_histories()
        self.writer = SummaryWriter(log_dir=pathlib.Path(cfg.OUTPUT_PATH).joinpath("runs/"))

    def _reset_histories(self):
        """Resets train and val histories for the accuracy and the loss. """

        self.train_loss_history = []
        self.train_acc_history = []
        self.train_dice_history = []
#         self.val_acc_history = []
#         self.val_loss_history = []
#         self.val_dice_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=5):
        """
        Train a given model with the provided data.

        Inputs:
        - model: object initialized from a torch.nn.Module
        - train_loader: train data (currently using nonsense data)
        - val_loader: val data (currently using nonsense data)
        - num_epochs: total number of epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        init_weights(model, init_type="normal")
        model.train()

        print("START TRAIN")
        start = time.time()

        for epoch in range(num_epochs):
            # Training
            for i, sample in enumerate(train_loader, 1):
                in_top, in_bottom = sample["top_image"].cuda().to(dtype=torch.float), \
                                    sample["bottom_image"].cuda().to(dtype=torch.float)
                gt_top, gt_bottom = sample["top_label"].cuda().to(dtype=torch.long), \
                                    sample["bottom_label"].cuda().to(dtype=torch.long)

                optim.zero_grad()

                out_top, out_bottom = model(in_top, in_bottom)
                loss = self.loss_func(out_top, out_bottom, gt_top, gt_bottom)
                loss.backward()
                optim.step()
                
                self.train_loss_history.append(loss.detach().cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN %s: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           self.loss_func.__name__,
                           train_loss))
                    self.writer.add_scalar("Loss", train_loss, i + epoch * iter_per_epoch)

            _, preds_top = torch.max(out_top, 1)
            _, preds_bottom = torch.max(out_bottom, 1)
            
            train_acc = np.mean((preds_top == gt_top.squeeze(1)).detach().cpu().numpy()) + \
                        np.mean((preds_bottom == gt_bottom.squeeze(1)).detach().cpu().numpy())
            train_acc /= 2
            
            train_dice = dice_coeff_2arm(out_top, out_bottom, gt_top, gt_bottom)
            
            self.train_acc_history.append(train_acc)
            self.train_dice_history.append(train_dice.detach().cpu().numpy())

            if log_nth:
                print('[Epoch %d/%d] TRAIN time/acc/dice/loss: %.3f/%.3f/%.3f/%.3f' %
                      (epoch + 1, num_epochs, time.time() - start, 
                       train_acc, train_dice, train_loss))

            # Validation
            val_losses = []
            val_scores = []
            val_dices = []
            model.eval()
            for j, sample in enumerate(val_loader, 1):
                in_top, in_bottom = sample["top_image"].cuda().to(dtype=torch.float), \
                                    sample["bottom_image"].cuda().to(dtype=torch.float)
                gt_top, gt_bottom = sample["top_label"].cuda().to(dtype=torch.long), \
                                    sample["bottom_label"].cuda().to(dtype=torch.long)

                out_top, out_bottom = model(in_top, in_bottom)
                loss = self.loss_func(out_top, out_bottom, gt_top, gt_bottom)
                val_losses.append(loss.detach().cpu().numpy())

                _, preds_top = torch.max(out_top, 1)
                _, preds_bottom = torch.max(out_bottom, 1)
                scores = np.mean((preds_top == gt_top.squeeze(1)).detach().cpu().numpy()) + \
                         np.mean((preds_bottom == gt_bottom.squeeze(1)).detach().cpu().numpy())
                scores /= 2
                val_scores.append(scores)
                
                val_dice = dice_coeff_2arm(out_top, out_bottom, gt_top, gt_bottom)
                val_dices.append(val_dice.detach().cpu().numpy())

            val_acc, val_loss, val_dice = np.mean(val_scores), np.mean(val_losses), np.mean(val_dices)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/dice/loss: %.3f/%.3f/%.3f' % (epoch + 1,
                                                                             num_epochs,
                                                                             val_acc,
                                                                             val_dice,
                                                                             val_loss))

            model.train()

        #################################################################

        end = time.time()
        print("FINISH")
        print("TIME ELAPSED: {0}".format(end - start))
