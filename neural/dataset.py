# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''Fits all extracted files from the MOUS dataset into a usable,
self-contained MEGDatasets structure.
It comprises:
-- torch datasets (train, valid, test)
-- useful information to map from PCA space (default: 40) to Sensor space (default: 273)
(e.g. scaling, meg mean, pca matrix)
'''

from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np
import torch as th
import tqdm
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset


class MegSubject(Dataset):
    def __init__(self, meg, forcings, length, subject_id):
        """
        Torch Dataset class for storing the stimuli and MEG response
        of a single subject.

        Inputs:
        - meg: shape [N, C, T] with N trials, C channels, T time steps
        - forcings: dict, stimulus features (e.g. word frequency)
                keys: word onset, length, freq, first_mask (?), last, first
                values: each is a [N, 1, T]
        - length: shape [], length T shared by all sequences
        - subject_id: int
        """
        self.meg = meg
        self.forcings = forcings
        self.length = length
        self.subject_id = subject_id

    def __len__(self):
        """Gives total number of samples N"""
        return self.meg.size(0)

    def __getitem__(self, idx):
        """Returns one sample of data"""
        return (self.meg[idx], {k: v[idx]
                                for k, v in self.forcings.items()}, self.length[idx],
                self.subject_id)


# Higher level object to store datasets with their information (vs. torch ConcatDataset)
MegDatasets = namedtuple("MegDataset",
                         "train_sets valid_sets test_sets meg_dim forcing_dims meg_scalers "
                         "pca_mats means")


def _narrow(tensor, indexes):
    return tensor.gather(2, indexes.expand(-1, tensor.size(1), -1))


def _prepare_forcing(forcing):
    '''Reformat forcing we extracted from the MOUS dataset.
    Inputs:
        - forcing: [N, T, 1] forcing feature (e.g. word frequency)
            Usually a value from the forcings dict, whose keys are:
            word onset, length, freq, n_before, n_after, first_mask
    '''
    forcing = th.from_numpy(forcing).float()

    # Reformat the forcing feature as [N, 1, T]
    if forcing.dim() == 2:
        forcing = forcing.unsqueeze(1)
    else:
        forcing = forcing.permute(0, 2, 1)

    # Normalize
    forcing_normalized = (forcing - forcing.mean()) / forcing.std()

    return forcing_normalized


def load_torch_megs(path, n_subjects_max=None, subject=None, init=60, exclude=[], include=[]):

    # Create dict to the paths of all extracted files (one per subject)
    path = Path(path)
    subjects = defaultdict(dict)
    for child in path.iterdir():
        if child.suffix == ".pth":  # e.g. meg_1076_4_visual.pth
            kind, sub, *_ = child.stem.split("_")
            subjects[sub][kind] = child

    # Select subjects of interest
    to_load = list(subjects.keys())
    to_load.sort()
    if subject is not None:
        to_load = [to_load[subject]]
    if n_subjects_max:
        to_load = to_load[:n_subjects_max]

    train_sets = []
    valid_sets = []
    test_sets = []
    meg_scalers = []
    means = []
    pca_mats = []

    iterator = tqdm.tqdm(to_load, leave=False, ncols=120, desc="Loading data...")

    # Loop over subjects of interest
    subjs = []
    for index, subject in enumerate(iterator):
        # Load meg and forcing extraction files
        megdata = th.load(subjects[subject]["meg"])
        forcings = th.load(subjects[subject]["forcing"])
        subjs.append(megdata.get("subject", subject))  # what does the second arg do?

        after = forcings.pop("word_n_after", None)
        before = forcings.pop("word_n_before", None)

        # Define new stimulus features (last_word, first_word)
        # from old stimulus features (word_n_after, word_n_before)
        # assuming they are inclusive
        if after is not None:
            forcings["last_word"] = (after == 1).astype(np.float32)
        if before is not None:
            forcings["first_word"] = (before == 1).astype(np.float32)
        if "is_stop" in forcings:
            last_word = "is_stop"
        else:
            last_word = "last_word"
        # Create mask to select the first stimulus only in the 2.5s epoch
        if "first_mask" not in forcings:
            stim = forcings["stimulus"]
            first = 0 * stim
            for row in range(len(stim)):
                low = 60
                start = low + stim[row, low:].nonzero()[0][0]
                end = (stim[row, start:] == 0).nonzero()[0]
                if len(end):
                    end = end[0]
                    first[row, start:start + end] = 1
                else:
                    # print(subject, row, stim[row])
                    first[row, start:] = 1
            forcings["first_mask"] = first

        # Include or exclude stimulus features based on their name (key of forcing dict)
        for name in exclude:
            if name not in forcings:
                raise ValueError(f"{name} is not a valid feature name.")
        for name in include:
            if name not in forcings:
                raise ValueError(f"{name} is not a valid feature name.")
        if include:
            feats = list(include)
        else:
            feats = list(forcings.keys())
        for name in exclude:
            feats.remove(name)

        forcings = {
            # just normalizes forcing and permutes to [N, 1, T]
            name: _prepare_forcing(forcing)
            for name, forcing in forcings.items() if name in feats
        }
        forcing_dims = {}
        for key, value in forcings.items():
            forcing_dims[key] = value.size(1)  # expected: 1

        meg = megdata["meg"]
        pca_mats.append(megdata["pca_mat"])
        if "meg_last_idx" in megdata:
            last_index = th.from_numpy(megdata["meg_last_idx"])
        else:
            last_index = th.full((meg.shape[0], ), meg.shape[1], dtype=th.long)

        # Scale (robust) meg data
        # TODO: separate scaling for each set (train, valid, test)?
        meg_scaler = RobustScaler()
        meg_scalers.append(meg_scaler)
        meg = meg_scaler.fit_transform(meg.reshape(-1, meg.shape[-1])).reshape(*meg.shape)
        meg = th.from_numpy(meg)

        # Remove trials where an amplitude is too high (e.g. 16) after scaling
        max_amplitude = meg.abs().max(dim=1)[0].max(dim=1)[0]
        mask = max_amplitude <= 16
        # print(mask.float().mean(), mask.shape)
        meg = meg[mask]
        forcings = {key: value[mask] for key, value in forcings.items()}
        last_index = last_index[mask]

        # Center meg data
        mean = meg.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        means.append(mean)
        meg = meg - mean

        # Change meg format: [N, T, C] -> [N, C, T]
        meg = meg.permute(0, 2, 1)
        meg_dim = meg.size(1)  # expected: C

        n_trials = meg.shape[0]

        # Separate trials into train / valid / test:
        # search for an end of sentence to do the cuts
        train, valid, test = 0.7, 0.1, 0.2

        for trial in range(int(train * n_trials),
                           int((train + valid) * n_trials)):
            if forcings[last_word][trial, 0, 60] > 0:  # end of sentence
                break
        idx_train = list(range(trial + 1))

        for trial in range(int((train + valid) * n_trials), n_trials):
            if forcings[last_word][trial, 0, 60] > 0:
                break
        idx_valid = list(range(idx_train[-1] + 1, trial + 1))
        idx_test = list(range(idx_valid[-1] + 1, n_trials))

        # Instantiate train/valid/test epoched datasets
        dataset_train = MegSubject(
            meg=meg[idx_train],
            forcings={k: v[idx_train]
                      for k, v in forcings.items()},
            length=1 + last_index[idx_train],
            subject_id=index)

        dataset_valid = MegSubject(
            meg=meg[idx_valid],
            forcings={k: v[idx_valid]
                      for k, v in forcings.items()},
            length=1 + last_index[idx_valid],
            subject_id=index)

        dataset_test = MegSubject(
            meg=meg[idx_test],
            forcings={k: v[idx_test]
                      for k, v in forcings.items()},
            length=1 + last_index[idx_test],
            subject_id=index)

        train_sets.append(dataset_train)
        valid_sets.append(dataset_valid)
        test_sets.append(dataset_test)

    print("subjects: ", subjs)
    print("Overall train size: ", sum(tr.meg.shape[0] for tr in train_sets))
    print("Overall valid size: ", sum(tr.meg.shape[0] for tr in valid_sets))
    print("Overall test size: ", sum(tr.meg.shape[0] for tr in test_sets))

    return MegDatasets(
        train_sets=train_sets,
        valid_sets=valid_sets,
        test_sets=test_sets,
        meg_scalers=meg_scalers,
        means=means,
        pca_mats=pca_mats,
        meg_dim=meg_dim,
        forcing_dims=forcing_dims)
