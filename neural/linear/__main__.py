# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# import libraries
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..dataset import load_torch_megs
from ..utils import get_metrics, inverse
from ..visuals import plt  # noqa
from .arx import ARX
from .receptive_field import RField
from .stats import report_correl


def get_parser():
    parser = argparse.ArgumentParser("lin", description="Train lin predictors using forcings")
    parser.add_argument(
        "-d", "--data", type=Path,
        required=True,
        help="Path to the data extracted.")
    parser.add_argument("--n-subjects", type=int, default=68, help="Max number of subjects")
    parser.add_argument("--out", type=Path, default=Path("dump"))
    parser.add_argument("--with-forcing", action="store_true", default=False)
    parser.add_argument("--with-init", action="store_true", default=False)
    parser.add_argument("--shuffle", action="store_true", default=False)
    parser.add_argument("--pca", type=int, help="Use PCA version of the data. "
                                                "Should be the dimension of the PCA used.")
    parser.add_argument("--n-workers", type=int, default=20, help="Workers for parallelization.")
    return parser


def make_repo_from_parser(args):
    '''Creates and outputs a generic results repository
    from the parser arguments.'''

    parts = ["linear"]
    name_args = dict(args.__dict__)
    ignore = ["data", "out"]
    for key in ignore:
        name_args.pop(key, None)
    for name, value in name_args.items():
        parts.append(f"{name}={value}")
    name = " ".join(parts)
    print(f"Experiment {name}")

    # data path
    if args.pca is not None:
        suffix = f"{args.pca}"
    else:
        suffix = "full"
    args.data = args.data / Path(suffix)
    print("Using dataset", args.data)

    # omar added
    name_reg = name + " reg"
    name_autoreg = name + " autoreg"

    out_reg = args.out / name_reg
    out_autoreg = args.out / name_autoreg

    out_reg.mkdir(exist_ok=True)
    out_autoreg.mkdir(exist_ok=True)
    return out_reg, out_autoreg


def permute_forcing(first_mask, forcing, permutation, init=60):
    initial = forcing[:, :, init:init + 1]
    mask = first_mask > 0
    print(mask.float().mean(), first_mask.min(), first_mask.max())
    return torch.where(mask, initial[permutation], forcing)


def shuffle_forcings(forcings, name):
    forcing = forcings[name]
    first_mask = forcings["first_mask"]
    permutation = torch.randperm(forcing.size(0))
    forcings[name] = permute_forcing(first_mask, forcing, permutation)


def eval_lin_models(subject,
                    data_path,
                    results_path_reg,
                    results_path_autoreg,
                    n_init=40,
                    tune_models=True,
                    with_init=True,
                    with_forcing=True,
                    shuffle=False):
    # Load dataset
    data = load_torch_megs(data_path, subject=subject)

    # Load the necessary to reverse PCA
    pca_mat = data.pca_mats[0]
    mean = data.means[0]
    scaler = data.meg_scalers[0]

    # Get train / valid / test sets, tensor shape [N, C, T]
    meg_train = data.train_sets[0].meg.numpy()
    meg_valid = data.valid_sets[0].meg.numpy()
    meg_test = data.test_sets[0].meg.numpy()

    forcing_keys = data.train_sets[0].forcings.keys()
    forcing_train = np.concatenate(list([data.train_sets[0].forcings[k]
                                        for k in forcing_keys]), axis=1)
    forcing_valid = np.concatenate(list([data.valid_sets[0].forcings[k]
                                        for k in forcing_keys]), axis=1)
    forcing_test = np.concatenate(list([data.test_sets[0].forcings[k]
                                        for k in forcing_keys]), axis=1)

    if not with_forcing:
        forcing_train = np.zeros_like(forcing_train)
        forcing_valid = np.zeros_like(forcing_valid)
        forcing_test = np.zeros_like(forcing_test)

    # Reformat [N, T, C]
    [meg_train, meg_valid, meg_test,
     forcing_train, forcing_valid, forcing_test] = [
        np.swapaxes(elem, 1, 2) for elem in [meg_train, meg_valid, meg_test,
                                             forcing_train, forcing_valid, forcing_test]
    ]

    ######################
    # LIN REG
    ######################

    # Instantiate
    rfield = RField(lag_u=260, penal_weight=1.8)

    # Tune hyperparameter on valid set
    alpha_scores = list()
    alphas = np.logspace(-3, 3, 5)

    if tune_models:

        for alpha in alphas:
            rfield.model.estimator = alpha
            rfield.fit(forcing_train, meg_train)
            meg_pred = rfield.predict(forcing_valid)
            meg_true = meg_valid
            # computing metrics
            alpha_score = get_metrics(meg_true, meg_pred)
            alpha_scores.append(alpha_score.mean())
        # plt.plot(np.log10(alphas), alpha_scores)
        # plt.ylabel('r')
        # plt.show()
        # plt.close()

        alpha = alphas[np.argmax(alpha_scores)]
        rfield.model.estimator = alpha

    # Retrain on train + valid set, save model
    rfield.fit(forcing_train, meg_train)
    torch.save(rfield, results_path_reg / f"model_trf_subject_{subject}.th")

    # Predict on test set
    meg_pred = rfield.predict(forcing_test)
    meg_true = meg_test

    # Reverse PCA
    meg_pred = inverse(mean, scaler, pca_mat, meg_pred)
    meg_true = inverse(mean, scaler, pca_mat, meg_true)

    # Save plot
    report_correl(meg_true, meg_pred, results_path_reg / "reg.png", 0)

    # Save prediction sample from all subjects
    torch.save({"meg_pred_epoch": meg_pred[0],
                "meg_true_epoch": meg_true[0],
                "meg_pred_evoked": meg_pred.mean(0),
                "meg_true_evoked": meg_true.mean(0)},
                results_path_reg / f"meg_prediction_subject_{subject}.th")

    # Compute metric (Pearson R)
    score_linreg = get_metrics(meg_true, meg_pred)

    # Permutation Feature Importance
    shuffled = {}
    to_shuffle = ["word_lengths", "word_freqs"] if shuffle else []
    for name in to_shuffle:

        # Permute forcing (via original torch forcing)
        forcing_test_torch = data.test_sets[0].forcings
        shuffle_forcings(forcing_test_torch, name)
        forcing_test_shuffle = np.concatenate(list([forcing_test_torch[k]
                                                    for k in forcing_keys]), axis=1)
        forcing_test_shuffle = np.swapaxes(forcing_test_shuffle, 1, 2)
        if not with_forcing:
            forcing_test_shuffle = np.zeros_like(forcing_test_shuffle)

        # Predict on test set
        meg_pred = rfield.predict(forcing_test_shuffle)
        meg_true = meg_test

        # Reverse pca
        meg_pred = inverse(mean, scaler, pca_mat, meg_pred)
        meg_true = inverse(mean, scaler, pca_mat, meg_true)

        # Compute metric (Pearson R)
        score_tmp = get_metrics(meg_true, meg_pred)
        shuffled[name] = score_tmp

    ######################
    # LIN AUTOREG
    ######################

    # Instantiate
    ridge = ARX(lag_u=n_init, lag_y=n_init, solver="ridge", penal_weight=1.8, scaling=False)

    # Tune hyperparameter on valid set
    alpha_scores = list()

    if tune_models:

        for alpha in alphas:
            ridge.penal_weight = alpha
            ridge.fit(forcing_train, meg_train)
            meg_init = np.zeros_like(meg_valid)
            if with_init:
                meg_init[:, :n_init, :] = meg_valid[:, :n_init, :]
            meg_pred = ridge.predict(
                forcing_valid, meg_init, start=n_init, eval="unrolled")
            meg_true = meg_valid
            # computing metrics
            alpha_score = get_metrics(meg_true, meg_pred)
            alpha_scores.append(alpha_score.mean())

        # plt.plot(np.log10(alphas), alpha_scores)
        # plt.ylabel('r')
        # plt.show()
        # plt.close()

        alpha = alphas[np.argmax(alpha_scores)]
        ridge.penal_weight = alpha

    # Retrain on train + valid set, save model
    ridge.fit(forcing_train, meg_train)
    torch.save(ridge, results_path_autoreg / f"model_rtrf_subject_{subject}.th")

    # Predict on test set
    meg_init = np.zeros_like(meg_test)
    if with_init:
        meg_init[:, :n_init, :] = meg_test[:, :n_init, :]
    meg_pred = ridge.predict(forcing_test, meg_init, start=n_init, eval="unrolled")
    meg_true = meg_test

    # Reverse PCA
    meg_pred = inverse(mean, scaler, pca_mat, meg_pred)
    meg_true = inverse(mean, scaler, pca_mat, meg_true)

    # Save plot
    report_correl(meg_true, meg_pred, results_path_autoreg / "autoreg.png", n_init)

    # Save prediction sample for all subjects
    torch.save({"meg_pred_epoch": meg_pred[0],
                "meg_true_epoch": meg_true[0],
                "meg_pred_evoked": meg_pred.mean(0),
                "meg_true_evoked": meg_true.mean(0)},
                results_path_autoreg / f"meg_prediction_subject_{subject}.th")

    # Compute metric (Pearson R)
    score_linautoreg = get_metrics(meg_true, meg_pred)

    # TODO: add Permutation Feature Importance for linear autoreg
    score_linautoreg = np.zeros_like(score_linreg)

    return score_linreg, score_linautoreg, shuffled


def main():

    # Make repository
    parser = get_parser()
    args = parser.parse_args()
    out_reg, out_autoreg = make_repo_from_parser(args)

    # Prepare model labels
    if (not args.with_init) and (args.with_forcing):
        label_add = "(no init)"
    elif (args.with_init) and (not args.with_forcing):
        label_add = "(no forcing)"
    elif (args.with_init) and (args.with_forcing):
        label_add = ""
    elif (not args.with_init) and (not args.with_forcing):
        label_add = "(no init, no forcing)"

    # Initialize result dicts
    reg_results = {"label": "lin reg " + label_add,
                   "scores": []}
    shuffled_results = {"word_freqs": [],
                        "word_lengths": []}
    autoreg_results = {"label": "lin autoreg " + label_add,
                       "scores": []}

    # Loop over subjects (in parallel)
    with ProcessPoolExecutor(args.n_workers) as pool:

        pendings = []
        for sub in range(args.n_subjects):
            pendings.append(
                pool.submit(
                    eval_lin_models,
                    sub,
                    args.data,
                    out_reg,
                    out_autoreg,
                    with_forcing=args.with_forcing,
                    with_init=args.with_init,
                    shuffle=args.shuffle))

        for pending in tqdm.tqdm(pendings):
            (score_linreg, score_linautoreg, shuffled) = pending.result()

            # stack results in lists
            reg_results["scores"].append(score_linreg)
            autoreg_results["scores"].append(score_linautoreg)
            for key, score_shuffled in shuffled.items():
                shuffled_results[key].append(score_shuffled)

    # Making numpy arrays from lists
    reg_results["scores"] = np.array(reg_results["scores"])
    autoreg_results["scores"] = np.array(autoreg_results["scores"])
    for key in shuffled_results.keys():
        shuffled_results[key] = np.array(shuffled_results[key])

    # # Converting to torch arrays
    # reg_results["scores"] = torch.from_numpy(reg_results["scores"])
    # autoreg_results["scores"] = torch.from_numpy(autoreg_results["scores"])
    # for key in shuffled_results.keys():
    #     shuffled_results[key] = torch.from_numpy(shuffled_results[key])

    # Save
    torch.save(reg_results, out_reg / "reference_metrics.th")
    torch.save(autoreg_results, out_autoreg / "reference_metrics.th")

    if args.shuffle:
        for key, value in shuffled_results.items():
            torch.save({'scores': value, 'label': 'lin reg ' + label_add},
                       out_reg / f"shuffled_{key}_metrics.th")


if __name__ == "__main__":
    main()
