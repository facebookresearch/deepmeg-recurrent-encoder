# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import pandas as pd

from .dataset import load_torch_megs
from .model import MegPredictor
from .train import train_eval_model
from .utils import get_metrics, inverse
from .visuals import report_correl


def get_parser():
    parser = argparse.ArgumentParser("neural", description="Train MEG predictor using forcings")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("dump"),
        help="Folder where checkpoints and metrics are saved.")
    parser.add_argument(
        "-R", "--restart", action='store_true', help='Restart training, ignoring previous run')
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Dataset related
    parser.add_argument(
        "-d", "--data", type=Path,
        required=True,
        help="Path to the data extracted.")
    parser.add_argument("-s", "--subjects", type=int, default=68,
                        help="Maximum number of subjects.")
    parser.add_argument("--pca", type=int, help="Use PCA version of the data. "
                                                "Should be the dimension of the PCA used.")
    parser.add_argument("-x", "--exclude", action="append", default=[], help="Exclude features")
    parser.add_argument("-i", "--include", action="append", default=[], help="Include features")

    # Optimization parameters
    parser.add_argument("-e", "--epochs", type=int, default=60,
                        help="Number of epochs to train for.")
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l1", action="store_true", help="Use L1 loss instead of MSE")

    # Parameters to the model
    parser.add_argument("--conv-layers", type=int, default=2,
                        help="Number of convolution layers in the encoder/decoder.")
    parser.add_argument("--lstm-layers", type=int, default=2,
                        help="Number of LSTM layers.")
    parser.add_argument("--conv-channels", type=int, default=512,
                        help="Output channels for convolutions.")
    parser.add_argument("--lstm-hidden", type=int, default=512,
                        help="Hidden dimension of the LSTM.")
    parser.add_argument("--subject-dim", type=int, default=16,
                        help="Dimension of the subject embedding.")

    # Other parameters
    parser.add_argument("--meg-init", type=int, default=40,
                        help="Number of MEG time steps to provide as basal state.")
    parser.add_argument("--no-forcings", action="store_false", dest="forcings", default=True,
                        help="Remove all forcings from the input.")
    parser.add_argument("--save-meg", action="store_true",
                        help="Save full MEG output for each subject.")

    return parser


def make_repo_from_parser(args, parser):
    args.out.mkdir(exist_ok=True)

    parts = []
    name_args = dict(args.__dict__)
    ignore = ["restart", "data", "out"]
    for key in ignore:
        name_args.pop(key, None)
    for name, value in name_args.items():
        if value != parser.get_default(name):
            if isinstance(value, Path):
                value = value.name
            elif isinstance(value, list):
                value = ",".join(map(str, value))
            parts.append(f"{name}={value}")
    if parts:
        name = " ".join(parts)
    else:
        name = "default"
    print(f"Experiment {name}")

    if args.pca is not None:
        suffix = f"{args.pca}"
    else:
        suffix = "full"

    # args.data = args.data.with_name(args.data.name + suffix)
    args.data = args.data / Path(suffix)
    print("Using dataset", args.data)

    out = args.out / name
    if args.restart and out.exists():
        shutil.rmtree(out)
    out.mkdir(exist_ok=True)

    return out


@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    state: dict = None
    best_state: dict = None


def main():

    # Make repository
    parser = get_parser()
    args = parser.parse_args()
    out = make_repo_from_parser(args, parser)

    # Set seed and device
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    th.manual_seed(args.seed)
    random.seed(args.seed)

    # Load data
    meg_sets = load_torch_megs(args.data, args.subjects, exclude=args.exclude, include=args.include)
    train_set = ConcatDataset(meg_sets.train_sets)
    valid_set = ConcatDataset(meg_sets.valid_sets)
    test_set = ConcatDataset(meg_sets.test_sets)

    # Instantiate model
    model = MegPredictor(
        meg_dim=meg_sets.meg_dim,
        forcing_dims=meg_sets.forcing_dims if args.forcings else {},
        meg_init=args.meg_init,
        subject_dim=args.subject_dim,
        conv_layers=args.conv_layers,
        conv_channels=args.conv_channels,
        lstm_layers=args.lstm_layers,
        lstm_hidden=args.lstm_hidden).to(device)

    # Instantiate optimization
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() if args.l1 else nn.MSELoss()
    train_eval = partial(
        train_eval_model,
        model=model,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        batch_size=args.batch_size)

    try:
        saved = th.load(out / "saved.th")
    except IOError:
        saved = SavedState()
    else:
        model.load_state_dict(saved.state)

    best_loss = float("inf")
    for epoch, metric in enumerate(saved.metrics):
        print(f"Epoch {epoch:04d}: "
              f"train={metric['train']:.4f} test={metric['valid']:.6f} best={metric['best']:.6f}")
        best_loss = metric['best']

    # Train and Evaluate (valid set) the model
    # from where you left off
    # select best model over the epochs on valid set`
    print("Training and Validation...")
    for epoch in range(len(saved.metrics), args.epochs):
        train_loss, _ = train_eval(train_set)
        with th.no_grad():
            valid_loss, evals = train_eval(valid_set, train=False, save=True)
        best_loss = min(valid_loss, best_loss)
        saved.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
        })
        print(f"Epoch {epoch:04d}: "
              f"train={train_loss:.4f} valid={valid_loss:.6f} best={best_loss:.6f}")
        if valid_loss == best_loss:
            saved.best_state = {
                key: value.to("cpu").clone()
                for key, value in model.state_dict().items()
            }
            th.save(model, out / "model.th")
            json.dump(saved.metrics, open(out / "metrics.json", "w"), indent=2)
        saved.state = {key: value.to("cpu") for key, value in model.state_dict().items()}
        th.save(saved, out / "saved.th")

    # Save train-valid curve
    tmp = pd.read_json(out / "metrics.json")
    fig, ax = plt.subplots()
    ax.plot(tmp['train'], color='red', label='train')
    ax.plot(tmp['valid'], color='green', label='valid')
    ax.set_ylabel('Train Loss')
    ax.legend()
    fig.savefig(out / "train_valid_curve.png")

    # Load best model (on the valid set)
    json.dump(saved.metrics, open(out / "metrics.json", "w"), indent=2)
    model.load_state_dict(saved.best_state)

    # Evaluate (test set) the model
    with th.no_grad():
        print("Evaluating model on test set...")

        # Reference evaluation
        ref_loss, ref_evals = train_eval(test_set, train=False, save=True)
        print("Ref loss", ref_loss)

        # Trim true and predicted meg
        # to a common time length (in case of an excess index)
        min_length = ref_evals.lengths.min().item()
        megs = ref_evals.megs[:, :, :min_length]
        ref_predictions = ref_evals.predictions[:, :, :min_length]

        # Reformat true and predicted meg: back to [N, T, C]
        megs = megs.transpose(1, 2)
        ref_predictions = ref_predictions.transpose(1, 2)

        # Loop over subjects
        ordered_subjects = ref_evals.subjects.unique().sort()[0]
        scores = list()

        for sub in ordered_subjects:
            sub_sel = (ref_evals.subjects == sub).nonzero().flatten()

            Y_true, Y_pred = megs[sub_sel], ref_predictions[sub_sel]

            # Load the necessary to reverse PCA
            pca_mat = meg_sets.pca_mats[sub]
            mean = meg_sets.means[sub]
            scaler = meg_sets.meg_scalers[sub]

            # Reverse PCA
            Y_true = inverse(mean, scaler, pca_mat, Y_true.double().numpy())
            Y_pred = inverse(mean, scaler, pca_mat, Y_pred.double().numpy())

            if args.save_meg:
                # Save prediction sample from all subjects [N, T, C]
                print(f"Saving MEG pred for sub {sub}...")
                th.save({"meg_pred_epoch": Y_pred[0],
                         "meg_true_epoch": Y_true[0],
                         "meg_pred_evoked": Y_pred.mean(0),
                         "meg_true_evoked": Y_true.mean(0)},
                        out / f"meg_prediction_subject_{sub}.th")

            # Correlation metrics between true and predicted meg, shape [N, T, C]
            score = get_metrics(Y_true, Y_pred)

            scores.append(score)

        scores = np.stack(scores)  # [S, T, C]
        print("Average prediction score (Pearson R): ", scores[:, 60:].mean())

        # Save results
        th.save({"scores": scores},
                out / "reference_metrics.th")

        # Shuffled-feature evaluations
        for name in list(meg_sets.forcing_dims) + ["meg", "subject"]:
            if name in ["word_onsets", "first_mask"]:
                continue
            test_loss, evals = train_eval(test_set, train=False, save=True, permut_feature=name)

            delta = (test_loss - ref_loss) / ref_loss
            print("Shuffled", name, "relative loss increase", 100 * delta)
            predictions = evals.predictions[:, :, :min_length].transpose(1, 2)
            assert (evals.megs == ref_evals.megs).all()
            report_correl(
                megs,
                predictions,
                out / f"feature_importance_{name}_correl_all.png",
                ref=ref_predictions,
                start=60)

            # Loop over subjects
            scores = list()

            for sub in ordered_subjects:
                sub_sel = (ref_evals.subjects == sub).nonzero().flatten()

                Y_true, Y_pred = megs[sub_sel], predictions[sub_sel]

                # Load the necessary to reverse PCA
                pca_mat = meg_sets.pca_mats[sub]
                mean = meg_sets.means[sub]
                scaler = meg_sets.meg_scalers[sub]

                # Reverse PCA
                Y_true = inverse(mean, scaler, pca_mat, Y_true.numpy())
                Y_pred = inverse(mean, scaler, pca_mat, Y_pred.numpy())

                # Correlation metrics
                score = get_metrics(Y_true, Y_pred)

                scores.append(score)

            scores = np.stack(scores)

            th.save({"scores": scores},
                    out / f"shuffled_{name}_metrics.th")


if __name__ == "__main__":
    main()
