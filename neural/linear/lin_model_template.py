# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries
import copy

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..visuals import plot_eigvalues, plt
from .stats import statespace_transform

# Mother class


class lin_model:
    def __init__(self, lag_y, lag_u, penal_weight=1., scaling=False, log=False):

        # model choice
        self.lag_u = lag_u
        self.lag_y = lag_y
        self.maxlag = np.max([self.lag_u, self.lag_y])
        self.scaling = scaling
        self.penal_weight = penal_weight

        # model architecture

        # learned feats
        self.weights = np.array([])
        self.weights_u = np.array([])
        self.weights_y = np.array([])
        self.A = np.array([])

        # data properties
        self.n_channels_y = 0
        self.n_channels_u = 0
        self.n_feats_x = 0
        self.n_feats_v = 0
        self.scaler_target = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_u = StandardScaler()

        self.solver = solver
        self.model = 0

        self.regressors_names = list()
        self.residuals = 0

        self.log = log

    def formulate_regression(self, U, Y):
        """
        Input:
        -----

        U : numpy array (n_samples x n_times x n_channels_u)
        Y : numpy array (n_samples x n_times x n_channels_y)
        """

        if self.log:
            print("----------------------------------------- \n")
            print("\n FORMULATING REGRESSION... \n")
            print("----------------------------------------- \n")

        # initializing variables
        n_samples, n_times, self.n_channels_y = Y.shape
        self.n_channels_u = U.shape[2]

        self.n_feats_x = self.n_channels_y * self.lag_y
        self.n_feats_v = self.n_channels_u * self.lag_u

        # scaling input timeseries
        if self.scaling:
            Y = self.scaler_y.fit_transform(Y.reshape(-1, Y.shape[-1]))
            U = self.scaler_u.fit_transform(U.reshape(-1, U.shape[-1]))
            Y = Y.reshape(n_samples, n_times, self.n_channels_y)
            U = U.reshape(n_samples, n_times, self.n_channels_u)

        # # zero padding for initialization?
        # U_ini = np.zeros((n_samples, self.lag_u, self.n_channels_u))
        # Y_ini = np.zeros((n_samples, self.lag_y, self.n_channels_y))
        # U = np.concatenate([U_ini, U], axis=1)
        # Y = np.concatenate([Y_ini, Y], axis=1)

        # convert: canonical space timeseries (Y, U) -> state space timeseries (X, V)
        X = statespace_transform(Y, self.lag_y)
        V = statespace_transform(U, self.lag_u)

        # take common time-length
        # Y and U having different lags and nb channels, their statespace timeseries have different lengths
        n_common_times = np.min([X.shape[1], V.shape[1]])
        X = X[:, -n_common_times:, :]
        V = V[:, -n_common_times:, :]
        _, _, n_feats_x = X.shape  # (n_samples, n_common_times, n_feats_x)
        _, _, n_feats_v = V.shape

        # formulate the problem as a regression
        regression_mat = np.concatenate(
            [
                X[:, :-1, :].reshape((n_common_times - 1) * n_samples, -1).T,  # ()
                V[:, 1:, :].reshape((n_common_times - 1) * n_samples, -1).T
            ],
            axis=0).T  # (n_common_times * n_samples, n_feats_x + n_feats_u)

        target = Y[:, -n_common_times + 1:, :].reshape(-1, self.n_channels_y)

        if self.log:
            print("shape of target: ", target.shape, "\n")
            print("shape of regression matrix: ", regression_mat.shape, "\n")

        return target, regression_mat

    def plot_weights(self, summarize=True, names_u=[]):

        if len(names_u) == 0:
            names_u = ["U Channel " + str(channel_u) for channel_u in range(self.n_channels_u)]

        if not summarize:

            # plot forcing weights
            fig, axes = plt.subplots(self.n_channels_u, 1, sharex=True)

            for channel_u in range(self.n_channels_u):

                axes[channel_u].set_title(names_u[channel_u] + " Weights")
                axes[channel_u].plot(self.weights_u[:, :, channel_u].T)

            plt.xlabel("Lags")
            plt.tight_layout()
            plt.show()
            plt.close()

            # plot recurrence weights
            # be careful: display optimized for even nb of MEG Princ Comps
            fig, axes = plt.subplots(self.n_channels_y // 2, 2, sharex=True)

            for channel_y in range(self.n_channels_y // 2):
                axes[channel_y, 0].set_title("Y Channel " + str(channel_y) + " Weights")
                axes[channel_y, 0].plot(self.weights_y[:, :, channel_y].T)

            for channel_y in range(self.n_channels_y // 2, self.n_channels_y):
                axes[channel_y - (self.n_channels_y // 2), 1].set_title("Y Channel " +
                                                                        str(channel_y) + " Weights")
                axes[channel_y - (self.n_channels_y // 2), 1].plot(
                    self.weights_y[:, :, channel_y].T)

            plt.xlabel("Lags")
            plt.tight_layout()
            plt.tight_layout()
            plt.show()
            plt.close()

        if summarize:

            fig, axes = plt.subplots(2, 1, figsize=(8.15, 3.53))

            # forcing weights
            for channel_u in range(self.n_channels_u):
                axes[0].fill_between(
                    range(self.lag_u),
                    np.mean(self.weights_u[:, :, channel_u]**2,
                            axis=0),  # mean over output channels
                    label=names_u[channel_u],
                    alpha=0.25)
                axes[0].set_title("Forcing Weights over Lags")
            axes[0].legend()

            # recurrence weights
            for channel_y in range(self.n_channels_y):
                axes[1].fill_between(
                    range(self.lag_y),
                    np.mean(self.weights_y[:, :, channel_y]**2,
                            axis=0),  # mean over output channels
                    label="channel " + str(channel_y),
                    alpha=0.25)
                axes[1].set_title("Recurrence Weights over Lags")

            plt.tight_layout()
            plt.show()
            plt.close()

    def plot_recurrence_eigvalues(self):

        plot_eigvalues(self.A)

    def predict(self, U=None, Y=None, start=0, eval="unrolled"):

        if self.log:
            print("\n-----------------------------------------")
            print("\n PREDICT \n")
            print("-----------------------------------------\n")

        # instantiate U and Y
        n_epochs, n_times, self.n_channels_u = U.shape

        if Y is None:
            Y = np.zeros((n_epochs, n_times, self.n_channels_y))
        if U is None:
            U = np.zeros((n_epochs, n_times, self.n_channels_u))

        # augment U and Y_pred with generic initializations for prediction
        U_ini = np.zeros((n_epochs, self.lag_u, self.n_channels_u))
        Y_ini = np.zeros((n_epochs, self.lag_y, self.n_channels_y))

        U_augmented = np.concatenate([U_ini, U], axis=1)
        Y_augmented = np.concatenate([Y_ini, Y], axis=1)

        if self.log:
            print(U_augmented.shape)
            print(Y_augmented.shape)

        # convert: canonical space timeseries -> state space timeseries
        V = statespace_transform(U_augmented, self.lag_u)
        X = statespace_transform(Y_augmented, self.lag_y)
        _, _, n_feats_v = V.shape
        _, _, n_feats_x = X.shape

        # make sure n_times + 1
        V = V[:, -(n_times + 1):, :]
        X = X[:, -(n_times + 1):, :]

        # standard scale
        # V_augmented = self.scaler_v.transform(V_augmented.reshape(-1, n_feats_v))
        # V_augmented = V_augmented.reshape(n_epochs, n_times + 1, n_feats_v)

        if self.log:
            print("\n Constructing predicted trajectories... \n")

        # initialize pred
        Y_pred = list()

        if start > 0:
            for idx in range(start):
                Y_pred.append(Y[:, idx, :])

        if eval == "onestep":
            X_true = copy.deepcopy(X)
            # THIS IS NEW
            # X = np.zeros(n_epochs, n_times, self.weights.shape[0])

        # print("DIM OF X_TRUE IS: ", X_true.shape)

        for t in range(start, n_times):

            # if self.scaling:
            #     V_contrib = np.array([(self.scaler_v.transform(V_augmented[epoch, t+1, :][None, :])).flatten()
            #                           for epoch in range(n_epochs)])
            #     X_contrib = np.array([(self.scaler_x.transform(X_pred_augmented[epoch, t, :][None, :])).flatten()
            #                           for epoch in range(n_epochs)])
            V_contrib = V[:, t + 1, :]

            if eval == "unrolled":
                X_contrib = X[:, t, :]
            elif eval == "onestep":
                X_contrib = X_true[:, t, :]

            # currstate = np.concatenate([X_pred_augmented[:, t, :],
            #                            V_augmented[:, t+1, :]],
            #                            axis=1)

            currstate = np.concatenate([X_contrib, V_contrib], axis=1)
            if self.log:
                print("dim of currstate: ", currstate.shape)
            # n_feats = currstate.shape[-1]

            A_reduced = self.weights  # (n_channels_y, n_feats)

            pred = currstate @ A_reduced.T
            if self.log:
                print("dim of pred: ", pred.shape)
            # if self.scaling:
            #     pred = np.array([(self.scaler_target.inverse_transform(pred[epoch][None, :])).flatten()
            #                     for epoch in range(n_epochs)])
            obj2 = X[:, t, :-self.n_channels_y:]
            if self.log:
                print("dim of obj2: ", obj2.shape)
            np.concatenate(
                [
                    pred,  # n_epochs, n_channels_y
                    obj2
                ],
                axis=1)  # concatenate over time
            if self.log:
                print("concatenation works!")
                print("dim of X[:, t+1, :]: ", X[:, t + 1, :].shape)
            X[:, t + 1, :] = np.concatenate(
                [
                    pred,  # n_epochs, n_channels_y
                    obj2
                ],
                axis=1)  # concatenate over time

        #     Y_pred.append(pred)

        # Y_pred = np.array(Y_pred)

        # X_pred = self.scaler_x.inverse_transform(X_pred)
        Y_pred = X[:, 1:, :self.n_channels_y]

        # return np.swapaxes(Y_pred, 0, 1)
        return Y_pred
