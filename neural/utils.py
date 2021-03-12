# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os.path
import shutil

import numpy as np


def create_directory(path, overwrite=False):

    # if it is not there, create it
    if not os.path.exists(path):
        os.makedirs(path)

    # if it is there and overwrite, remove then recreate
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
        os.makedirs(path)

    return 0


def get_metrics(Y_true, Y_pred):
    '''Computes the correlation of two [B, T, C] tensors
    over the first dimension.
    In this case, yields epoch-wise correlation per
    time step and channel.

    Inputs:
    - Y_true: torch tensor [N, T, C], truth
    - Y_pred: torch tensor [N, T, C], prediction

    Output:
    - R_matrix: torch tensor [T, C] of Pearson R scores
    '''
    dim = 0  # avg-out epochs

    Y_true = Y_true - Y_true.mean(axis=dim, keepdims=True)
    Y_pred = Y_pred - Y_pred.mean(axis=dim, keepdims=True)
    cov = (Y_true * Y_pred).mean(dim)
    na, nb = [(i**2).mean(dim)**0.5 for i in [Y_true, Y_pred]]
    norms = na * nb
    R_matrix = cov / norms  # shape (T, C)

    return R_matrix


def center_trim(tensor, reference):
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def inverse(mean, scaler, pca, Y):
    pca_channels, full_channels = pca.shape
    trials, time, channels = Y.shape

    Y = Y + mean.numpy()
    Y = Y.reshape(-1, channels)
    Y = scaler.inverse_transform(Y)
    return np.reshape(Y @ pca, (trials, time, full_channels))
