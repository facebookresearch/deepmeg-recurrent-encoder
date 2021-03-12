# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries
import numpy as np
from mne.decoding import ReceptiveField

from ..visuals import plt


class RField:
    def __init__(self, lag_u, penal_weight=1e3):
        self.lag_u = lag_u
        self.penal_weight = penal_weight
        self.model = ReceptiveField(
            tmin=0., tmax=lag_u, sfreq=1., estimator=self.penal_weight)
        self.n_channels_u = 0

    def fit(self, U, Y):

        self.n_channels_u = U.shape[2]
        # swap 2 first axes for MNE:
        # (n_samples, n_times, n_channels) -> (n_times, n_samples, n_channels)
        self.model.fit(np.swapaxes(U, 0, 1), np.swapaxes(Y, 0, 1))

    def plot_weights(self, summarize=True, names_u=[]):

        if len(names_u) == 0:
            names_u = [
                "U Channel " + str(channel_u)
                for channel_u in range(self.n_channels_u)
            ]

        if not summarize:

            # plot forcing weights
            fig, axes = plt.subplots(self.n_channels_u, 1, sharex=True)

            for increment, channel_u in enumerate(
                    list(range(self.n_channels_u))):

                axes[channel_u].set_title(names_u[channel_u] + " Weights")
                weights = self.model.coef_[:, channel_u, :].T
                axes[channel_u].plot(weights)

            plt.xlabel("Lags")
            plt.tight_layout()
            plt.show()
            plt.close()

        if summarize:

            fig, axes = plt.subplots(2, 1, figsize=(10, 5))

            # forcing weights
            for increment, channel_u in enumerate(
                    list(range(self.n_channels_u))):
                weights = self.model.coef_[:, channel_u, :].T
                axes[0].fill_between(
                    range(weights.shape[0]),
                    np.sum(weights**2, axis=1),
                    label=names_u[channel_u],
                    alpha=0.25)
                axes[0].set_title("Forcing Weights over Lags")
            axes[0].legend()

            plt.tight_layout()
            plt.show()
            plt.close()

    def predict(self, U, U_ini=np.array([]), Y_ini=np.array([])):

        return np.swapaxes(self.model.predict(np.swapaxes(U, 0, 1)), 0, 1)
