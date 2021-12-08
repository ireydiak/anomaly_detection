import argparse
from collections import defaultdict

import torch

from src.datamanager.image import CIFAR10Manager
from src.exp import BatchTrainer, Experiment
from src.image.models import ALAD, DeepSVDD
from src.image.trainers import ALADTrainer, DeepSVDDTrainer
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
    # args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Selected device %s' % device)

    #print(f"Training model {args.model} on {args.dataset}")

    #nep_mode = args.neptune_mode
    #nep_tags = [args.dataset, args.model]
    # Experiment's params
    # path_to_dataset: str, export_path: str, device: str,
    # neptune_mode: str, neptune_tags: list, dataset: str, pct: float, model: str, epochs: int, n_runs: int,
    # batch_size: int, model_param=None, seed=None
    # exp = Experiment(
    #     args.dataset_path, args.output_path, device, nep_mode, nep_tags, args.dataset,
    #     args.pct, args.model, args.epochs, args.n_runs, args.batch_size, {}, None
    # )
    # trainer = BatchTrainer([exp])
    # print(trainer.summary())
    # trainer.train()
    # Setting up the model, trainer and their respective parameters
    n_epochs = 200
    batch_size = 32
    latent_dim = 32
    lr = 1e-4
    device = 'cuda'
    seed = 42
    n_runs = 1
    normal_class = 'cat'

    ds = CIFAR10Manager(root='./data', normal_class=normal_class)
    model = ALAD(feature_dim=3, latent_dim=latent_dim)
    trainer = ALADTrainer(model=model, n_epochs=n_epochs, batch_size=batch_size, lr=lr, device='cuda')

    # Training
    # (neptune) initialize neptune
    # run = load_neptune(exp.neptune_mode, exp.neptune_tags)
    run = None
    # Load train_set, test_set, trainer and model
    # ds = resolve_dataset(exp.dataset, exp.path_to_dataset, exp.pct)
    # TODO: load params from exp

    train_ldr, test_ldr = ds.loaders(batch_size=batch_size, seed=seed)

    # (neptune) log datasets sizes
    # run["data/train/size"] = len(train_ldr)
    # run["data/test/size"] = len(test_ldr)

    # (neptune) set parameters to be uploaded
    # tau = exp.tau or int(np.ceil((1 - ds.anomaly_ratio) * 100))
    tau = int(np.ceil((1 - ds.anomaly_ratio) * 100))
    # all_params = {
    #     "N": ds.N,
    #     "runs": n_runs,
    #     "anomaly_threshold": tau,
    #     "anomaly_ratio": "%1.4f" % ds.anomaly_ratio,
    #     **trainer.get_params()
    # }
    # run["parameters"] = all_params

    training_start_time = dt.now()
    print("Training %s with shape %s, anomaly ratio %1.4f" % ("DeepSVDD", ds.shape, ds.anomaly_ratio))
    P, R, FS, ROC, PR = [], [], [], [], []
    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs}")
        metrics = train_once(trainer, train_ldr, test_ldr, tau, run)
        print(metrics)
        for m, val in zip([P, R, FS, ROC, PR], metrics.values()):
            m.append(val)
        model.reset()
    P, R, FS, ROC, PR = np.array(P), np.array(R), np.array(FS), np.array(ROC), np.array(PR)
    res = defaultdict()
    for vals, key in zip([P, R, FS, ROC, PR], ["Precision", "Recall", "F1-Score", "AUROC", "AUPR"]):
        res[key] = "%2.4f (%2.4f)" % (vals.mean(), vals.std())
    print(res)
