import argparse
from collections import defaultdict

import torch

from src.datamanager import IEEFraudDetection
from src.exp import Experiment, BatchTrainer
from src.tabular.models import DeepSVDD
from src.tabular.trainers import DeepSVDDTrainer
import numpy as np
from datetime import datetime as dt

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model',
    help='The selected model',
    choices=['DeepSVDD', 'MemAE', 'DAGMM', 'DSEBM-e', 'DSEBM-r', 'NeuTralAD'],
    type=str
)
parser.add_argument(
    '-d', '--dataset',
    help='The selected dataset',
    choices=['KDD10', 'NSLKDD', 'USBIDS', 'Arrhythmia', 'IDS2017', 'Thyroid'],
    type=str
)
parser.add_argument(
    '-b', '--batch-size',
    help='Size of mini-batch',
    default=128,
    type=int
)
parser.add_argument(
    '-e', '--epochs',
    help='Number of training epochs',
    default=100,
    type=int
)
parser.add_argument(
    '-o', '--output-path',
    help='Path to the output folder',
    default=None,
    type=str
)
parser.add_argument(
    '--n-runs',
    help='Number of time model is trained',
    default=20,
    type=int
)
parser.add_argument(
    '--tau',
    help='Threshold beyond which samples are labeled as anomalies.',
    default=None,
    type=int
)
parser.add_argument(
    '-p', '--dataset-path',
    help='Path to the dataset (set when --timeout-params is empty)',
    type=str,
    default=None
)
parser.add_argument(
    '--timeout-params',
    help='Hyphen-separated timeout parameters (FlowTimeout-Activity Timeout)',
    type=str,
    nargs='+'
)
parser.add_argument(
    '--seed',
    help='Specify a seed',
    type=int,
    default=None
)
parser.add_argument(
    '--pct',
    help='Percentage of original data to keep',
    type=float,
    default=1.
)
parser.add_argument(
    '--neptune-mode',
    help='Use Neptune.ai to track training metadata',
    type=str,
    default='async'
)


def train_once(trainer, train_ldr, test_ldr, tau, nep):
    trainer.train(train_ldr, nep)
    y_train_true, train_scores = trainer.test(train_ldr)
    y_test_true, test_scores = trainer.test(test_ldr)
    y_true = np.concatenate((y_train_true, y_test_true), axis=0)
    scores = np.concatenate((train_scores, test_scores), axis=0)
    print("Evaluating model with threshold tau=%d" % tau)
    return trainer.evaluate(y_true, scores, threshold=tau)


if __name__ == '__main__':
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Selected device %s' % device)
    print(f"Training model {args.model} on {args.dataset}")

    nep_mode = args.neptune_mode
    nep_tags = [args.dataset, args.model]
    # Experiment's params
    exp = Experiment(
        args.dataset_path, args.output_path, device, nep_mode, nep_tags, args.dataset,
        args.pct, args.model, args.epochs, args.n_runs, args.batch_size, {}, None
    )
    trainer = BatchTrainer([exp])
    print(trainer.summary())
    trainer.train()
