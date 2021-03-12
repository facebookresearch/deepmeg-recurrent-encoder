# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .lin_model_template import lin_model
from .stats import quick_svd


class ARX(lin_model):
    def __init__(self, lag_u, lag_y, solver="ridge", penal_weight=1., scaling=False, log=False):

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

    def fit(self, U, Y):
        """
        Input:
        -----

        U : numpy array (n_samples x n_times x n_channels_u)
        Y : numpy array (n_samples x n_times x n_channels_y)
        """

        if self.log:
            print("----------------------------------------- \n")
            print("\n TRAIN \n")
            print("----------------------------------------- \n")

        target, regression_mat = self.formulate_regression(U, Y)

        # plug-in model to solve the regression
        if self.log:
            print("Solving the Least-Squares...\n")
        if self.solver == "ridge":

            self.model = Ridge(alpha=self.penal_weight)
            self.model.fit(regression_mat, target)
            weights = self.model.coef_.T
            residuals = target - regression_mat @ weights
            if self.log:
                print("shape of weights: ", weights.T.shape, "\n")

        if self.solver == "feasible":  # using statsmodels

            self.model = sm.GLSAR(target, regression_mat)
            result = self.model.fit()
            weights = result.params

            if len(weights.shape) == 1:  # fixing dimensionality bug
                weights = weights.reshape(weights.shape[-1], -1)

            residuals = target - regression_mat @ weights
            if self.log:
                print("shape of weights: ", weights.T.shape, "\n")

        if self.solver == "dmd":  # pseudo-inverse the tranposed system

            # formulation
            # (n_feats_x + n_feats_u, n_common_times * n_samples)
            snapshots = regression_mat.T
            snapshots_next = target.T
            if self.log:
                print("shape of snapshot matrix: ", snapshots.shape)

            # compute surrogate right-side svd
            UU, DD, VVt = quick_svd(snapshots, rank_perc=self.penal_weight)

            # inverse the system and obtain weights
            # left-side svd (optional? not done here)
            if self.log:
                print("\n Inverting the system... \n")

            weights = snapshots_next @ VVt.T @ np.diag(1. / DD) @ UU.T
            weights = weights.T
            residuals = target - regression_mat @ weights
            if self.log:
                print("shape of weights: ", weights.T.shape)

        # record learnt coefficients
        self.residuals = residuals
        self.weights = weights.T
        if len(self.weights.shape) == 0:  # debug
            self.weights = self.weights[None, :]

        # reformat weights to [(n_output_channels, n_lags, n_input_channels)
        # for u, for y]
        self.weights_y = self.weights[:, :self.lag_y * self.n_channels_y].reshape(
            self.n_channels_y, self.lag_y, self.n_channels_y)
        self.weights_u = self.weights[:, self.lag_y * self.n_channels_y:].reshape(
            self.n_channels_y, self.lag_u, self.n_channels_u)

        # form recurrence matrix from weights
        if self.lag_y != 0:
            self.A = np.concatenate([
                self.weights_y.reshape(self.n_channels_y, -1),
                np.eye(N=self.n_feats_x - self.n_channels_y, M=self.n_feats_x, k=self.n_channels_y)
            ],
                                    axis=0)
        else:
            self.A = np.zeros((self.n_feats_x, self.n_feats_x))
