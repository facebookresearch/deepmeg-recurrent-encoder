# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch as th
from torch import nn
from torch.nn import functional as F

from .utils import center_trim


class MegPredictor(nn.Module):
    def __init__(self,
                 meg_dim,
                 forcing_dims,
                 meg_init=40,
                 n_subjects=100,
                 max_length=301,
                 subject_dim=16,
                 conv_layers=2,
                 kernel=4,
                 stride=2,
                 conv_channels=256,
                 lstm_hidden=256,
                 lstm_layers=2):
        super().__init__()
        self.forcing_dims = dict(forcing_dims)
        self.meg_init = meg_init

        in_channels = meg_dim + 1 + subject_dim + sum(forcing_dims.values())

        if subject_dim:
            self.subject_embedding = nn.Embedding(n_subjects, subject_dim)
        else:
            self.subject_embedding = None

        channels = conv_channels
        encoder = []
        for _ in range(conv_layers):
            encoder += [
                nn.Conv1d(in_channels, channels, kernel, stride, padding=kernel // 2),
                nn.ReLU(),
            ]
            in_channels = channels
        self.encoder = nn.Sequential(*encoder)
        if lstm_layers:
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers)
            in_channels = lstm_hidden
        else:
            self.lstm = None
        self.conv_layers = conv_layers
        self.stride = stride
        self.kernel = kernel
        if conv_layers == 0:
            self.decoder = nn.Conv1d(in_channels, meg_dim, 1)
        else:
            decoder = []
            for index in range(conv_layers):
                if index == conv_layers - 1:
                    channels = meg_dim
                decoder += [
                    nn.ConvTranspose1d(in_channels, channels, kernel, stride, padding=kernel // 2),
                ]
                if index < conv_layers - 1:
                    decoder += [nn.ReLU()]
                in_channels = channels
            self.decoder = nn.Sequential(*decoder)

    def get_meg_mask(self, meg, forcings):
        batch, _, time = meg.size()
        mask = th.zeros(batch, 1, time, device=meg.device)
        mask[:, :, :self.meg_init] = 1.
        return mask

    def valid_length(self, length):
        for _ in range(self.conv_layers):
            length = math.ceil(length / self.stride) + 1
        for _ in range(self.conv_layers):
            length = (length - 1) * self.stride
        return int(length)

    def pad(self, x):
        length = x.size(-1)
        valid_length = self.valid_length(length)
        delta = valid_length - length
        return F.pad(x, (delta // 2, delta - delta // 2))

    def forward(self, meg, forcings, subject_id):
        forcings = dict(forcings)
        batch, _, length = meg.size()
        inputs = []

        mask = self.get_meg_mask(meg, forcings)
        meg = meg * mask
        inputs += [meg, mask]

        if self.subject_embedding is not None:
            subject = self.subject_embedding(subject_id)
            inputs.append(subject.view(batch, -1, 1).expand(-1, -1, length))

        if self.forcing_dims:
            _, forcings = zip(*sorted([(k, v)
                                       for k, v in forcings.items() if k in self.forcing_dims]))
        else:
            forcings = {}

        inputs.extend(forcings)

        x = th.cat(inputs, dim=1)
        x = self.pad(x)
        x = self.encoder(x)
        if self.lstm is not None:
            x = x.permute(2, 0, 1)
            x, _ = self.lstm(x)
            x = x.permute(1, 2, 0)
        out = self.decoder(x)
        return center_trim(out, length)
