# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import seaborn as sns
import torch


def statespace_transform(Y, lag):
    """
    Converts canonical-space timeseries into state-space timeseries
    by squeezing multichannel past, up to the chosen lag order,
    into a single lag vector.

    The advantage of using the state-space, is being able to write any
    finite-dimensional Dynamical System as order 1.

    https://en.wikipedia.org/wiki/Vector_autoregression#Writing_VAR(p)_as_VAR(1)

    Input:
    -----

    Y : numpy array, (n_samples, n_times, n_channels)
        Observed timeseries.

    lag: float
        Chosen autoregression order.

    Output:
    ------

    Y_statespace : numpy array, (n_samples, n_times_statespace, n_feats_statespace)
        Statespace timeseries, where:
        n_times_statespace = n_times - lag
        n_feats_statespace = n_channels * lag

    """

    # print("\n Converting to Statespace... \n")

    # Initialize
    n_samples, n_times, _ = Y.shape
    Y_statespace = []

    for sample in range(n_samples):

        # We want most recent on top = at the beginning in statespace vector
        Y_statespace.append(
            np.array([
                Y[sample, np.arange(t, t + lag)[::-1], :].flatten()
                for t in range(n_times - lag + 1)
            ]))

    return np.array(Y_statespace)


def statespace_inversetransform(Y_statespace, n_channels, lag):
    """
    Input:
    -----

    Y_statespace : numpy array, (n_samples, n_times_statespace, n_feats_statespace)
        Statespace timeseries, where:
        n_times_statespace = n_times - lag
        n_feats_statespace = n_channels * lag

    n_channels: int
               Nb of original channels.

    lag: float
        Chosen autoregression order.


    Output:
    -----

    Y : numpy array, (n_samples, n_times, n_channels)
        Observed timeseries.
    """

    # print("\n From Statespace to Original space... \n")

    n_samples = Y_statespace.shape[0]
    Y = []

    for sample in range(n_samples):

        Y.append(
            np.concatenate([
                Y_statespace[sample, 0, :].reshape(lag, n_channels)[::-1],
                Y_statespace[sample, 1:, :n_channels]
            ],
                           axis=0))

    return np.array(Y)


# format for LSTM: (B, T, C) -> (T, B, C) in pytorch
def LSTM_format_transform(data, device):
    """data is a numpy array of shape (B, T, C).
    We want to output a torch tensor of shape (T, B, C)."""
    return torch.from_numpy(data).float().transpose(0, 1)


def LSTM_format_inversetransform(data):
    """data is a numpy array of shape (T, B, C).
    We want to output a torch tensor of shape (B, T, C)."""
    return data.transpose(0, 1).cpu().detach().numpy()


def svd_fat(X, rank_perc=1.):

    "Quick truncated svd for horizontal matrix."

    sym = X @ X.T

    eigvectors, eigvalues, _ = scipy.linalg.svd(sym)
    eigvalues = np.real(eigvalues)

    # trim it
    picks = np.arange(int(rank_perc * eigvalues.size))
    # picks = np.where(eigvalues > truncate)[0]
    U = eigvectors[:, picks]
    D = np.sqrt(eigvalues[picks])
    temp = np.diag(1. / D) @ U.T
    Vt = temp @ X

    return U, D, Vt


def quick_svd(X, rank_perc=1.):
    """Quick, approximate, truncated svd for matrix depending on its
    horizontal/vertical ratio."""

    n_lines, n_cols = X.shape

    # fat matrix
    if n_cols > 2 * n_lines:
        # print("\n Computing SVD using fat matrix option... \n")
        U, D, Vt = svd_fat(X, rank_perc)
        return U, D, Vt

    # tall matrix
    if n_lines > 2 * n_cols:
        # print("\n Computing SVD using tall matrix option... \n")
        U, D, Vt = svd_fat(X.T, rank_perc)
        return Vt.T, D, U.T

    # squarish matrix
    else:
        # print("\n Computing SVD directly using scipy function... \n")
        U, D, Vt = scipy.linalg.svd(X)

        # truncate
        D = np.real(D)
        picks = np.arange(int(rank_perc * D.size))
        # picks = np.where(D > truncate)[0]
        U = U[:, picks]
        D = D[picks]
        Vt = Vt[picks]

        return U, D, Vt


def R_score(Y_true, Y_pred, avg_out="times"):
    """
    Y_true: ndarray (n_epochs, n_times, n_channels_y)
    Y_pred: ndarray (n_epochs, n_times, n_channels_y)
    """

    Y_true = torch.from_numpy(Y_true)
    Y_pred = torch.from_numpy(Y_pred)

    if avg_out == "epochs":
        dim = 0

    elif avg_out == "times":
        dim = 1

    cov = (Y_true * Y_pred).mean(dim)
    na, nb = [(i**2).mean(dim)**0.5 for i in [Y_true, Y_pred]]
    norms = na * nb
    R_matrix = cov / norms
    return R_matrix.mean(1).cpu().numpy()  # avg over channels


def R_score_v2(Y_true, Y_pred, score="r", avg_out="times", start=0):
    """
    Y_true: numpy or torch (B, T, C)
    Y_pred: numpy or torch (B, T, C)
    """

    if type(Y_true) is not np.ndarray:
        Y_true = torch.from_numpy(Y_true)
    if type(Y_pred) is not np.ndarray:
        Y_pred = torch.from_numpy(Y_pred)

    if avg_out == "epochs":
        dim = 0

    elif avg_out == "times":
        dim = 1
        Y_pred, Y_true = Y_pred[:, start:, :], Y_true[:, start:, :]

    if score == "r":
        cov = (Y_true * Y_pred).mean(dim)
        na, nb = [(i**2).mean(dim)**0.5 for i in [Y_true, Y_pred]]
        norms = na * nb
        R_matrix = cov / norms

    if score == "relativemse":
        Y_err = Y_pred - Y_true
        R_matrix = (Y_err**2).mean(dim) / (Y_true**2).mean(dim)  # rename this score matrix!!!

    if type(R_matrix) is not np.ndarray:
        R_matrix = R_matrix.cpu().numpy()

    return R_matrix


def report_correl(Y_true, Y_pred, path, start):
    """
    Y_true: ndarray (n_epochs, n_times, n_channels_y)
    Y_pred: ndarray (n_epochs, n_times, n_channels_y)
    """

    r_dynamic_epochs = R_score_v2(Y_true, Y_pred, avg_out="epochs")

    r_average_epochs = R_score_v2(Y_true, Y_true.mean(0, keepdims=True), avg_out="epochs")

    mse_dynamic_epochs = R_score_v2(Y_true, Y_pred, score="relativemse", avg_out="epochs")

    mse_average_epochs = R_score_v2(
        Y_true, Y_true.mean(0, keepdims=True), score="relativemse", avg_out="epochs")
    # r_scalar = R_score(Y_true,
    #                    Y_pred,
    #                    avg_out="times").mean()

    r_average_times = R_score_v2(Y_true[:, start:, :], Y_pred[:, start:, :], avg_out="times")

    r_average_times_evoked = R_score_v2(
        Y_true[:, start:, :].mean(0, keepdims=True),
        Y_pred[:, start:, :].mean(0, keepdims=True),
        avg_out="times")

    mse_average_times = R_score_v2(
        Y_true[:, start:, :], Y_pred[:, start:, :], score="relativemse", avg_out="times")

    mse_average_times_evoked = R_score_v2(
        Y_true[:, start:, :].mean(0, keepdims=True),
        Y_pred[:, start:, :].mean(0, keepdims=True),
        score="relativemse",
        avg_out="times")

    # print(r_scalar)

    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    # Mean response
    axes[0, 0].plot(Y_pred.mean(0))
    axes[0, 0].set_title("Predicted Response (Evoked)")
    axes[0, 0].axvline(x=start, ls="--")
    axes[0, 0].text(x=start, y=0, s="init")

    axes[1, 0].plot(Y_true.mean(0))
    axes[1, 0].set_title("True Response (Evoked)")

    # Reponse to one stimulus
    axes[0, 1].plot(Y_pred[0])
    axes[0, 1].set_title("Predicted Response (Epoch 0)")
    axes[0, 1].axvline(start, ls="--")
    axes[0, 1].text(x=start, y=0, s="init")

    axes[1, 1].plot(Y_true[0])
    axes[1, 1].set_title("True Response (Epoch 0)")

    # Dynamic Correlation score
    axes[0, 2].plot(r_dynamic_epochs.mean(-1), label="epoch-wise correlation")
    # axes[0, 2].plot(r_dynamic_evoked, label="evoked-wise correlation")
    axes[0, 2].plot(r_average_epochs.mean(-1), label="baseline correlation")
    axes[0, 2].legend()
    axes[0, 2].set_title("Correlation along time")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].locator_params(axis='x', nbins=20)
    axes[0, 2].locator_params(axis='y', nbins=10)
    axes[0, 2].grid()
    axes[0, 2].axvline(start, ls="--")
    axes[0, 2].text(x=start, y=0, s="init")

    # Scalar Correlation score
    # axes[1, 2].bar([0, 1, 2], [0, r_scalar, 0])
    # axes[1, 2].set_title("Correlation Score")

    # Distributional Correlation score
    # epoched
    scores = r_average_times.T.flatten()
    pca_labels = np.concatenate(
        [[idx] * r_average_times.shape[0] for idx in range(r_average_times.shape[1])])
    df = pd.DataFrame({"scores": scores, "pca_labels": pca_labels})
    sns.boxplot(x="pca_labels", y="scores", data=df, ax=axes[1, 2])
    axes[1, 2].set_title("Overall Correlation")
    # evoked
    scores = r_average_times_evoked.mean(0)
    pca_labels = np.arange(r_average_times.shape[-1])
    axes[1, 2].plot(pca_labels, scores, label="corr of the trial-mean")
    axes[1, 2].legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)

    # Dynamic MSE score
    axes[0, 3].plot(mse_dynamic_epochs.mean(-1), label="epoch-wise mse")
    # axes[0, 2].plot(r_dynamic_evoked, label="evoked-wise correlation")
    axes[0, 3].plot(mse_average_epochs.mean(-1), label="baseline mse")
    axes[0, 3].legend()
    axes[0, 3].set_title("Relative MSE along time")
    axes[0, 3].set_ylim(0, 1)
    axes[0, 3].locator_params(axis='x', nbins=20)
    axes[0, 3].locator_params(axis='y', nbins=10)
    axes[0, 3].grid()
    axes[0, 3].axvline(start, ls="--")
    axes[0, 3].text(x=start, y=0, s="init")

    # Distributional MSE score
    # epoched
    scores = mse_average_times.T.flatten()
    pca_labels = np.concatenate(
        [[idx] * mse_average_times.shape[0] for idx in range(mse_average_times.shape[1])])
    df = pd.DataFrame({"scores": scores, "pca_labels": pca_labels})
    sns.boxplot(x="pca_labels", y="scores", data=df, ax=axes[1, 3])
    axes[1, 3].set_title("Overall Relative MSE")
    # evoked
    scores = mse_average_times_evoked.mean(0)
    pca_labels = np.arange(mse_average_times.shape[-1])
    axes[1, 3].plot(pca_labels, scores, label="rel. MSE of the trial-mean")
    axes[1, 3].legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def report_correl_all(Y_trues, Y_preds, U_trues, path):
    """
    Y_trues: list of ndarray (n_epochs, n_times, n_channels_y)
    Y_preds: list of ndarray (n_epochs, n_times, n_channels_y)
    U_trues: list of ndarray (n_epochs, n_times, n_channels_u)
    """

    rs_dynamic_epochs = list()
    rs_dynamic_evoked = list()
    rs_scalar = list()

    for subject, example in enumerate(zip(Y_trues, Y_preds, U_trues)):

        # unpack
        Y_true, Y_pred, U_true = example

        # calculate metrics for a given subject
        r_dynamic_epochs = R_score(Y_true, Y_pred, avg_out="epochs")

        r_dynamic_evoked = R_score(
            Y_true.mean(0, keepdims=True), Y_pred.mean(0, keepdims=True), avg_out="epochs")

        r_scalar = R_score(Y_true, Y_pred, avg_out="times").mean()

        # record these metrics
        rs_dynamic_epochs.append(r_dynamic_epochs)
        rs_dynamic_evoked.append(r_dynamic_evoked)
        rs_scalar.append(r_scalar)

    r_dynamic_epochs_mean = np.mean(rs_dynamic_epochs, axis=0)
    r_dynamic_epochs_std = np.std(rs_dynamic_epochs, axis=0)

    r_dynamic_evoked_mean = np.mean(rs_dynamic_evoked, axis=0)
    r_dynamic_evoked_std = np.std(rs_dynamic_evoked, axis=0)

    r_scalar_mean = np.mean(rs_scalar)
    r_scalar_std = np.std(rs_scalar)

    fig, axes = plt.subplots(3, 3, figsize=(15, 5))

    # Mean response for a subject
    axes[0, 0].plot(Y_preds[0].mean(0))
    axes[0, 0].set_title("A Predicted Response (Evoked)")

    axes[1, 0].plot(Y_trues[0].mean(0))
    axes[1, 0].set_title("A True Response (Evoked)")

    axes[2, 0].plot(U_trues[0].mean(0)[:, 0], label="word presence")
    axes[2, 0].set_title("Stimulus (only onset shown here)")
    axes[2, 0].legend()

    # Reponse to one stimulus for a subject
    axes[0, 1].plot(Y_preds[0][0])
    axes[0, 1].set_title("A Predicted Response (Epoch 0)")

    axes[1, 1].plot(Y_trues[0][0])
    axes[1, 1].set_title("A True Response (Epoch 0)")

    axes[2, 1].plot(U_trues[0][0][:, 0], label="word presence")
    axes[2, 1].plot(U_trues[0][0][:, 1], label="word length")
    axes[2, 1].plot(U_trues[0][0][:, 2], label="word frequency")
    axes[2, 1].set_title("A Stimulus (all features)")
    axes[2, 1].legend(loc="upper right")

    # Dynamic Correlation score
    axes[0, 2].plot(r_dynamic_epochs_mean, label="epoch-wise correlation", color="#B03A2E")
    axes[0, 2].fill_between(
        range(r_dynamic_epochs_mean.size),
        r_dynamic_epochs_mean - r_dynamic_epochs_std,
        r_dynamic_epochs + r_dynamic_epochs_std,
        color="#F1948A",
        alpha=0.5)

    axes[0, 2].plot(r_dynamic_evoked_mean, label="evoked-wise correlation", color="#2874A6")
    axes[0, 2].fill_between(
        range(r_dynamic_evoked_mean.size),
        r_dynamic_evoked_mean - r_dynamic_evoked_std,
        r_dynamic_evoked_mean + r_dynamic_evoked_std,
        color="#85C1E9",
        alpha=0.5)

    axes[0, 2].legend()
    axes[0, 2].set_title("Correlation along time")

    # Scalar Correlation score
    axes[1, 2].bar([0, 1, 2], [0, r_scalar_mean, 0], yerr=[0, r_scalar_std, 0])
    axes[1, 2].set_title("Correlation Score")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def report_correl_across_models(Y_trues, Y_preds, names, path):
    """
    Y_trues: list of ndarray (n_epochs, n_times, n_channels_y)
    Y_preds: list of ndarray (n_epochs, n_times, n_channels_y)
    """

    rs_dynamic_epochs = list()
    rs_scalar = list()

    for Y_true, Y_pred in zip(Y_trues, Y_preds):

        # calculate metrics for a given subject
        r_dynamic_epochs = R_score(Y_true, Y_pred, avg_out="epochs")

        r_scalar = R_score(Y_true, Y_pred, avg_out="times").mean()

        # record these metrics
        rs_dynamic_epochs.append(r_dynamic_epochs)
        rs_scalar.append(r_scalar)

    # correlation (trial-wise)
    fig, axes = plt.subplots(2, 1)

    for idx, name in enumerate(names):
        axes[0].plot(rs_dynamic_epochs[idx], label=name)

    axes[0].set_xlabel("Time steps")
    axes[0].set_title("Correlation (trial-wise)")
    axes[0].legend()

    # correlation (time-wise)
    axes[1].barh(range(len(names)), rs_scalar)
    axes[1].set_yticks(np.arange(len(names)))
    axes[1].set_yticklabels(names)
    axes[1].set_title("Correlation (time-wise)")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
