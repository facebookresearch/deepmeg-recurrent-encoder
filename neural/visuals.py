# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Largely unused.
Functions for visualization. '''

import os

import matplotlib
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns


# Ugly hack because my editor reorder imports automatically
def init():
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    return plt


def R_score_v2(Y_true, Y_pred, score="r", avg_out="times"):
    """
    Y_true: ndarray (n_epochs, n_times, n_channels_y)
    Y_pred: ndarray (n_epochs, n_times, n_channels_y)
    """

    if avg_out == "epochs":
        dim = 0

    elif avg_out == "times":
        dim = 1

    if score == "r":
        Y_true = Y_true - Y_true.mean(dim, keepdim=True)
        Y_pred = Y_pred - Y_pred.mean(dim, keepdim=True)
        cov = (Y_true * Y_pred).mean(dim)
        na, nb = [(i**2).mean(dim)**0.5 for i in [Y_true, Y_pred]]
        norms = na * nb
        R_matrix = cov / norms

    if score == "relativemse":
        Y_err = Y_pred - Y_true
        R_matrix = (Y_err**2).mean(dim) / (Y_true**2).mean(dim)  # rename this score matrix!!!

    return R_matrix.cpu().numpy()


def report_correl(Y_true, Y_pred, path, start, ref=None):
    """
    Y_true: ndarray (n_epochs, n_times, n_channels_y)
    Y_pred: ndarray (n_epochs, n_times, n_channels_y)
    """

    r_dynamic_epochs = R_score_v2(Y_true, Y_pred, avg_out="epochs")

    if ref is not None:
        r_average_epochs = R_score_v2(Y_true, ref, avg_out="epochs")
        ratio = (r_dynamic_epochs / r_average_epochs).mean(-1)
    else:
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
    if ref is not None:
        # axes[0, 2].plot(
        #     -np.log10(1e-8 + np.clip(1 - ratio, 0, 1)),
        #     label="log10 1 - ratio of correl")
        axes[0, 2].plot(ratio)
    else:
        axes[0, 2].plot(r_dynamic_epochs.mean(-1), label="epoch-wise correlation")
        axes[0, 2].plot(r_average_epochs.mean(-1), label="baseline correlation")
        axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].set_title("Correlation along time")
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


def make_train_test_curve(train_losses, test_losses, path, show=False, save=True):
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.gca().locator_params(axis='x', nbins=20)
    plt.gca().locator_params(axis='y', nbins=10)
    plt.gca().grid()
    plt.xlabel("Nb epochs")
    plt.ylabel("MSE")
    plt.title("Losses")
    plt.legend()
    if show:
        plt.show()
    if save:
        plt.savefig(path)
    plt.close()


def plot_eigvalues(A, add_to_title=""):

    # get eigenvalues
    eigs = scipy.linalg.eigvals(A)

    # plotting the eigenvalues
    plt.plot(eigs.real, eigs.imag, 'o')
    plt.plot(eigs.real, eigs.imag, 'rx')

    # plot unit circle
    thetas = np.linspace(0, 2 * np.pi)
    plt.plot(np.cos(thetas), np.sin(thetas), ls='--', c='gray')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axis('equal')

    # add stability thresholds
    plt.axvline(-1, linestyle="--")
    plt.axvline(1, linestyle="--")
    plt.text(1, 5, "stability threshold", rotation=90, verticalalignment='center')
    plt.text(-1, 5, "stability threshold", rotation=90, verticalalignment='center')

    plt.title(add_to_title + "\n" + "Recurrence Matrix Eigenvalues")
    plt.show()
    plt.close()


def plot_score_per_time(scores_per_time, labels, sfreq, ref=None, path=None, title=None):
    """
        input:
        -- scores_per_time : list of score_per_time arrays of shape (S, T, C)
        -- labels: list of labels of same len as scores
        -- ref: reference score_per_time, as in an upper bound
        """

    n_models = len(scores_per_time)
    n_subjects, n_times, n_channels = scores_per_time[0].shape

    # convert times to ms
    times = (np.arange(n_times) / sfreq) * 1000

    for idx in range(n_models):

        # current elements
        score_per_time = scores_per_time[idx].mean(-1)
        if ref is not None:
            score_per_time = ref.mean(-1) - score_per_time
        label = labels[idx]

        # take mean and SEM error over subjects
        score_per_time_mean = score_per_time.mean(0)
        score_per_time_error = score_per_time.std(0) / np.sqrt(n_subjects)

        # plot
        plt.plot(times, score_per_time_mean, label=label)
        plt.fill_between(
            times,
            y1=score_per_time_mean + score_per_time_error,
            y2=score_per_time_mean - score_per_time_error,
            alpha=0.5)

    plt.legend()
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    else:
        plt.title("Sample-wise Correlation between predicted and truth")
    plt.xlabel("Time (ms)")

    if path is not None:
        plt.savefig(os.path.join(path, "scores_per_time.png"))


def plot_score_per_time_topo(scores_per_time,
                             labels,
                             info,
                             sfreq=120,
                             ref=None,
                             path=None,
                             title=None):
    """
        input:
        -- scores_per_time : list of score_per_time arrays of shape (S, T, C)
        -- labels: list of labels of same len as scores
        -- ref: reference score_per_time, as in an upper bound
        """

    n_models = len(scores_per_time)
    n_subjects, n_times, n_channels = scores_per_time[0].shape

    fig, axes = plt.subplots(n_models, 2)

    for idx in range(n_models):

        # current elements
        score_per_time = scores_per_time[idx]
        if ref is not None:
            score_per_time = ref - score_per_time
        label = labels[idx]

        # take mean over subjects
        evo_data = score_per_time.mean(0).T  # (C, T)
        evo = mne.EvokedArray(evo_data, info=info, tmin=-.500)

        # plot time course
        mne.viz.plot_evoked(
            evo,
            spatial_colors=True,
            scalings=dict(mag=1.),
            show=False,
            axes=axes[idx, 0],
            titles='')
        # ax[0].set_ylim(-.01, .11)
        axes[idx, 0].set_xlabel('time')
        axes[idx, 0].set_ylabel('$\\Delta{}r$')
        axes[idx, 0].set_title('Feature %s' % label)

        # plot topo
        vmax = evo_data.mean(1).max()
        im, _ = mne.viz.plot_topomap(
            evo_data.mean(1),
            evo.info,
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
            show=False,
            axes=axes[idx, 1])

        plt.colorbar(im, ax=axes[idx, 1])

    plt.tight_layout()

    if path is not None:
        plt.savefig(path / "sensors_feature_importance.pdf")


def plot_score(scores, labels, ref=None, path=None, title=None):
    """
        input:
        -- scores : list of scores arrays of shape (S,)
        -- labels: list of labels of same len as scores
        """

    n_models = len(scores)

    bps = []  # list of boxplots

    for idx in range(n_models):

        # current elements
        score = scores[idx]
        if ref is not None:
            score = ref - score

        bp = plt.boxplot(score, positions=[idx])
        bps.append(bp)

    plt.xlim(-0.5, n_models + 1)
    plt.legend([bp["boxes"][0] for bp in bps], [label for label in labels], loc='upper right')
    if title is not None:
        plt.title(title)
    else:
        plt.title("Temporal Correlation between predicted and truth")
    plt.tight_layout()

    if path is not None:
        plt.savefig(os.path.join(path, "scores.png"))


plt = init()
