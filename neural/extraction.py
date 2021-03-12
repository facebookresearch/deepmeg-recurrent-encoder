# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries

from concurrent.futures import ProcessPoolExecutor
import argparse
import os
import traceback

from sklearn.decomposition import PCA
import numpy as np
import mne
import torch as th

from .utils_mous import (
    create_directory, get_word_length, get_word_freq,
    read_log, get_log_times, _add_stim_id, setup_logfiles, setup_stimuli)

mne.set_log_level(False)


def get_parser():
    '''Parser, to input arguments from the terminal.'''
    parser = argparse.ArgumentParser("extraction",
                                     description="Extract meg and forcing from MOUS Dataset")
    parser.add_argument("--data", type=str, help="Path to MOUS dataset")
    parser.add_argument("--out", type=str, help="Path where to save output")
    parser.add_argument("--use-pca", action="store_true", default=False)
    parser.add_argument("--pca-dim", type=int, default=40)
    parser.add_argument("--n-subjects", type=int, default=-1,
                        help="Maximum number of subjects to extract from.")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers.")
    return parser


def make_repo_from_parser(args, parser):
    '''Creates a repository whose name is based on the arguments of the parser.'''
    output_repo = args.out
    if args.use_pca:
        output_repo += f"/{args.pca_dim}"
    else:
        output_repo += "/full"
    create_directory(output_repo)

    return output_repo


# collect arguments and make repository
parser = get_parser()
args = parser.parse_args()
output_directory = make_repo_from_parser(args, parser)
print("output directory: ", output_directory)
data_path = args.data

# create legible csv tables from MOUS data for meg and stimuli
cache = "./cache"
create_directory(cache)
log_files = setup_logfiles(data_path, cache)
stimuli = setup_stimuli(data_path, cache)

# select files for given task
log_files = log_files.query('task=="visual"').reset_index(drop=True)


def extract_subject(subject):
    '''Extracts MEG and forcing for a subject in the MOUS Dataset.

    Input:
        subject (int): subject identifier
    '''
    log_file = log_files.iloc[subject]
    try:
        # generic output filename
        output_fname = [
            "%s",
            str(log_file["subject"]),
            str(log_file["log_id"]), log_file["task"]
        ]
        output_fname = "_".join(output_fname) + ".pth"

        ##########################
        # LOAD MEG AND LOG
        ##########################

        # get meg and log filenames
        raw_fname = os.path.join(data_path, log_file['meg_file'])
        log_fname = os.path.join(data_path, log_file['log_file'])

        # read meg (continuous)
        raw = mne.io.read_raw_ctf(raw_fname, preload=True)
        raw.filter(1., 30.)  # Slow

        # preprocess annotations and add task information
        log = read_log(log_fname, stimuli)
        log = _add_stim_id(log, verbose=False, stimuli=stimuli)  # get words

        # adding n words before and after in sentence
        log_words = log.query('condition=="word"')
        words_idx = log.query('condition=="word"').index

        sentence_lengths = np.bincount(log_words.sequence_pos.values.astype(int))

        n_words_before = np.concatenate([np.arange(length) + 1
                                         for length in sentence_lengths]).flatten()
        n_words_before = n_words_before.astype(int)

        n_words_after = np.concatenate([np.ones(length) * length
                                       for length in sentence_lengths]).flatten() \
                                       - n_words_before
        n_words_after = n_words_after.astype(int)

        log.loc[words_idx, "n_words_before"] = n_words_before
        log.loc[words_idx, "n_words_after"] = n_words_after

        # find events
        events = mne.find_events(raw, min_duration=.010)
        # link meg and annotations
        log = get_log_times(log, events, raw.info['sfreq'])

        ##########################
        # EXTRACT MEG
        ##########################

        # select desired event
        log_events = log.query('condition=="word"')

        # format events for mne
        log_events_formatted = np.c_[log_events.meg_sample,
                                     np.ones((len(log_events), 2), int)]
        _, idx = np.unique(log_events_formatted[:, 0], return_index=True)

        # segment meg into word-locked epochs
        picks = mne.pick_types(raw.info,
                               meg=True,
                               eeg=False,
                               stim=False,
                               eog=False,
                               ecg=False)
        decim = 10
        tmin, tmax = -.500, 2

        epochs = mne.Epochs(
            raw,
            events=log_events_formatted,
            metadata=log_events,
            tmin=tmin,
            tmax=tmax,
            decim=decim,
            preload=True,
            picks=picks,
        )

        # throw away compensation channels
        bads = [epochs.ch_names[i] for i in range(28)]  # hardcoded
        raw = raw.pick_types(meg=True, exclude=bads)
        epochs = epochs.pick_types(meg=True, exclude=bads)

        # get evoked meg
        evoked = epochs.average(method='mean')

        # get pca on evoked
        evoked_temp = evoked.apply_baseline().data.T * 1e12  # scaled
        duration_for_pca = int((np.abs(tmin) + 1) * epochs.info["sfreq"])
        evoked_temp = evoked_temp[:duration_for_pca]  # cropped
        if args.use_pca:
            pca = PCA(args.pca_dim).fit(evoked_temp)
            pca_mat = pca.components_
        else:
            pca_mat = np.eye(evoked_temp.shape[1], dtype=np.float32)

        ##########################
        # SAVE MEG
        ##########################

        # collect
        meg = epochs.get_data()
        meg_evoked = evoked.apply_baseline().data[None, :, :]

        # useful for sentences of different lengths
        meg_last_idx = (np.abs(tmin) + tmax) * epochs.info["sfreq"] * np.ones(len(epochs))
        meg_last_idx = meg_last_idx.astype(int)

        # reformat
        meg = np.swapaxes(meg, 1, 2)
        meg_evoked = np.swapaxes(meg_evoked, 1, 2)

        meg_pca = meg @ pca_mat.T
        times = np.array(epochs.metadata["time"], dtype=np.float32)

        # save
        output_dict = dict(
            zip(["meg", "meg_last_idx", "pca_mat", "epochs_info", "times", "subject"],
                [meg_pca.astype(np.float32),
                 meg_last_idx, pca_mat, epochs.info, times, log_files.subject[subject]
        ]))
        output_path = os.path.join(output_directory, output_fname % "meg")
        print("output path: ", output_path)
        th.save(output_dict, output_path)

        ##########################
        # LOAD FORCING
        ##########################

        n_epochs, n_channels, n_times = epochs.get_data().shape
        forcing_word = np.zeros((n_epochs, 6, n_times), dtype=np.float32)

        for epo_idx in range(n_epochs):

            # continuous time interval
            on = epochs.metadata.iloc[epo_idx].time
            start, end = on - np.abs(tmin), on + tmax

            # corresponding words
            cond = (start < epochs.metadata.time) & (epochs.metadata.time < end)

            words = epochs.metadata[cond].word.values.flatten().tolist()

            # recentering the time interval around the main onset
            onsets = epochs.metadata[cond].time.values - on + np.abs(tmin)

            # converting the time interval from s to Tsampl
            onsets = (onsets * epochs.info["sfreq"]).astype(int)

            # recovering word durations, then offsets
            durations = epochs.metadata[cond].Duration.values.astype(float) * 1e-4  # unit: second
            durations = (durations * epochs.info["sfreq"]).astype(int)  # unit: time sample
            offsets = onsets + durations

            # getting features
            word_lengths = get_word_length(words)
            word_freqs = get_word_freq(words)
            word_n_before = epochs.metadata[cond].n_words_before.values.flatten().tolist()
            # add + 1 to make difference with no forcing
            word_n_after = (epochs.metadata[cond].n_words_after.values.flatten() + 1).tolist()

            # placing square on word presence
            for idx, (onset, offset) in enumerate(zip(onsets, offsets)):
                forcing_word[epo_idx, 0, onset: offset] = 1.
                forcing_word[epo_idx, 1, onset:offset] = word_lengths[idx]
                forcing_word[epo_idx, 2, onset:offset] = word_freqs[idx]
                forcing_word[epo_idx, 3, onset:offset] = word_n_before[idx]
                forcing_word[epo_idx, 4, onset:offset] = word_n_after[idx]
                if idx == 0:
                    # mask for first forcing used to shuffle features
                    forcing_word[epo_idx, 5, onset:offset] = 1.

        # save forcing
        forcing_names = ["word_onsets", "word_lengths", "word_freqs",
                         "word_n_before", "word_n_after", "first_mask"]

        forcing = [forcing_word[:, 0, :][:, None, :],
                   forcing_word[:, 1, :][:, None, :],
                   forcing_word[:, 2, :][:, None, :],
                   forcing_word[:, 3, :][:, None, :],
                   forcing_word[:, 4, :][:, None, :],
                   forcing_word[:, 5, :][:, None, :],
                   ]

        # reformat
        forcing = [np.swapaxes(f, 1, 2) for f in forcing]

        # save
        output_dict = dict(zip(forcing_names, forcing))
        output_path = os.path.join(output_directory,
                                   output_fname % "forcing")
        th.save(output_dict, output_path)

    except Exception as e:
        print(f"Error {e} with subject {subject} {log_file}")
        traceback.print_exc()
        return
    else:
        print("SUBJECT", subject, "done")


# loop over subjects
if args.n_subjects == -1:
    n_subjects = len(log_files)
else:
    n_subjects = args.n_subjects
if args.workers == 1:
    for subject in range(n_subjects):
        extract_subject(subject)
else:
    with ProcessPoolExecutor(args.workers) as pool:
        for subject in range(n_subjects):
            pool.submit(extract_subject, subject)
