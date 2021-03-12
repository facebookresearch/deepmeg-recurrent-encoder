# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Trains a model with a train_eval_model function.'''

from collections import namedtuple

import torch as th
from torch import nn
from torch.utils import data
from tqdm import tqdm

SavedEval = namedtuple("SavedEval", "megs forcings predictions lengths subjects")


def permute_forcing(first_mask, forcing, permutation, init=60):
    initial = forcing[:, :, init:init + 1]
    mask = first_mask > 0
    return th.where(mask, initial[permutation], forcing)


def train_eval_model(dataset,
                     model,
                     optimizer=None,
                     progress=True,
                     train=True,
                     save=False,
                     device="cpu",
                     batch_size=128,
                     permut_feature=None,
                     criterion=nn.MSELoss()):
    '''Train and Eval function.

    Inputs:
    ...
    - save: if True, the second output is a namedtuple with
            megs, [N, C, T]
            forcings, dict of values [N, 1, T]
            predictions, [N, C, T]  # if regularization, T can change
            lengths, [N]
            subjects, [N]
    '''

    dataloaded = data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    if train:
        desc = "train set"
        model.train()
    else:
        desc = "test set"
        model.eval()

    running_loss = 0

    dl_iter = iter(dataloaded)
    if progress:
        dl_iter = tqdm(dataloaded, leave=False, ncols=120, total=len(dataloaded), desc=desc)

    if save:
        saved = SavedEval([], [], [], [], [])

    batch_idx = 0

    for batch_idx, batch in enumerate(dl_iter):

        # Unpack batch and load unto device (e.g. gpu)
        meg, forcings, length, subject_id = batch  # [B, C, T], dict of values [B, C, 1], [B], [B]
        meg = meg.to(device)
        forcings = {k: v.to(device) for k, v in forcings.items()}
        subject_id = subject_id.to(device)
        true_subject_id = subject_id
        length = length.to(device)

        n_batches, channels, n_times = meg.size()

        meg_true = meg

        # Permute an input feature (to measure its importance at test time)
        if permut_feature is not None:
            permutation = th.randperm(n_batches, device=device)
            if permut_feature == "meg":
                permutation = permutation.view(-1, 1, 1).expand(-1, meg.size(1), meg.size(-1))
                meg = th.gather(meg, 0, permutation)
            elif permut_feature == "subject":
                subject_id = th.gather(subject_id, 0, permutation)
            else:
                forcing = forcings[permut_feature]
                forcings[permut_feature] = permute_forcing(forcings["first_mask"], forcing,
                                                           permutation)
        saved_forcings = forcings

        # Predict, evaluate loss, backprop
        meg_pred = model(meg, forcings, subject_id)
        loss_train = criterion(meg_pred, meg_true)
        loss = criterion(meg_pred[..., model.meg_init:], meg_true[..., model.meg_init:])
        running_loss += loss.item()

        if train:
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()

        if save:
            # all quantities (meg, forcings, length, subject_id) saved in their original state,
            # except forcing which is saved in its permuted state
            saved.megs.append(meg_true.cpu())
            saved.forcings.append({k: v.cpu() for k, v in saved_forcings.items()})
            saved.predictions.append(meg_pred.detach().cpu())
            saved.lengths.append(length.cpu())
            saved.subjects.append(true_subject_id.cpu())
        if progress:
            dl_iter.set_postfix(loss=running_loss / (batch_idx + 1))

    n_batches = batch_idx + 1  # idx starts at 0
    running_loss /= n_batches  # average over batches

    if save:
        saved = SavedEval(
            megs=th.cat(saved.megs),
            forcings={k: th.cat([v[k] for v in saved.forcings])
                      for k in forcings},
            predictions=th.cat(saved.predictions),
            lengths=th.cat(saved.lengths),
            subjects=th.cat(saved.subjects))
    else:
        saved = None

    return running_loss, saved
