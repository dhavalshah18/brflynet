"""Class to do training of the network."""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from .misc import init_weights
from .misc import compute_loss


class SolverBtrfly(object):
    default_optim_args = {"lr": 0.01,
                          "weight_decay": 0.}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=compute_loss):

        optim_args_merged = self.default_optim_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.best_train_model = None
        self.best_val_model = None

        self._reset_histories()
        self.writer = SummaryWriter()

    def _reset_histories(self):
        """Resets train and val histories for the accuracy and the loss. """

        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
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
                gt_top, gt_bottom = sample["top_label"].cuda().to(dtype=torch.float), \
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
                    print('[Iteration %d/%d] TRAIN loss: %.3f' %
                          (i + epoch * iter_per_epoch,
                           iter_per_epoch * num_epochs,
                           train_loss))
                    self.writer.add_scalar("Loss", train_loss, i + epoch * iter_per_epoch)

            # _, preds = torch.max(outputs, 1)
            # train_acc = np.mean((preds == targets).detach().cpu().numpy())
            # self.train_acc_history.append(train_acc)
            # self.train_dice_coeff_history.append(dice_coeff)

            if log_nth:
                # print('[Epoch %d/%d] TRAIN time/acc/loss: %.3f/%.3f/%.3f' %
                #       (epoch + 1, num_epochs, time.time() - start, train_acc, train_loss))
                print('[Epoch %d/%d] TRAIN time/loss: %.3f/%.3f' %
                      (epoch + 1, num_epochs, time.time() - start, train_loss))

            # Validation
            val_losses = []
            val_scores = []
            model.eval()
            for j, sample in enumerate(val_loader, 1):
                in_top, in_bottom = sample["top_image"].cuda().to(dtype=torch.float), \
                                    sample["bottom_image"].cuda().to(dtype=torch.float)
                gt_top, gt_bottom = sample["top_label"].cuda().to(dtype=torch.float), \
                                    sample["bottom_label"].cuda().to(dtype=torch.long)

                out_top, out_bottom = model(in_top, in_bottom)
                loss = self.loss_func(out_top, out_bottom, gt_top, gt_bottom)
                val_losses.append(loss.detach().cpu().numpy())

                # _, preds = torch.max(outputs, 1)
                # scores = np.mean((preds == targets).detach().cpu().numpy())
                # val_scores.append(scores)

            # val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            val_loss = np.mean(val_losses)
            if log_nth:
                # print('[Epoch %d/%d] VAL   acc/loss/dice: %.3f/%.3f/%.3f' % (epoch + 1,
                #                                                              num_epochs,
                #                                                              val_acc,
                #                                                              val_loss, dice_coeff))
                print('[Epoch %d/%d] VAL   loss: %.3f' % (epoch + 1, num_epochs, val_loss))

            model.train()

        #################################################################

        end = time.time()
        print("FINISH")
        print("TIME ELAPSED: {0}".format(end - start))
