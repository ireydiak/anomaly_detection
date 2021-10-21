import argparse
from collections import defaultdict
from datetime import datetime as dt
import numpy as np
import torch
import os
from src.datamanager.dataset import *
from src.trainers.DeepSVDDTrainer import DeepSVDDTrainer
from src.models.OneClass import DeepSVDD

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='The selected model', choices=['DeepSVDD'], type=str)
parser.add_argument('-d', '--dataset', help='The selected dataset', choices=['KDD10', 'NSLKDD', 'USBIDS', 'Arrhythmia', 'IDS2017'], type=str)
parser.add_argument('-b', '--batch-size', help='Size of mini-batch', default=128, type=int)
parser.add_argument('-e', '--epochs', help='Number of training epochs', default=100, type=int)
parser.add_argument('-o', '--output-path', help='Path to the output folder', default=None, type=str)
parser.add_argument('--n-runs', help='Number of time model is trained', default=20, type=int)
parser.add_argument('--tau', help='Threshold beyond which samples are labeled as anomalies.', default=None, type=int)
parser.add_argument('--pct', help='Percentage of original data to keep', default=1., type=float)
parser.add_argument('--timeout-params', help='Hyphen-separated timeout parameters (FlowTimeout-Activity Timeout)', type=str, nargs='+')


def resolve_dataset(dataset_name: str, path_to_dataset: str, pct: float) -> AbstractDataset:
    clsname = globals()[f'{dataset_name}Dataset']
    return clsname(path_to_dataset, pct)


def store_results(results: dict, params: dict, model_name: str, dataset: str, path: str, start_time: dt, output_path: str=None):
    output_dir = output_path if output_path else f'./results/{dataset}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_dir + '/' + f'{model_name}_results.txt', 'a') as f:
        hdr = "Experiments on {}-{}\n".format(
            start_time.strftime("%d/%m/%Y %H:%M:%S"), dt.now().strftime("%d/%m/%Y %H:%M:%S")
        )
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")


def train_once(trainer, train_ldr, test_ldr, tau):
    trainer.train(train_ldr)
    y_train_true, train_scores = trainer.test(train_ldr)
    y_test_true, test_scores = trainer.test(test_ldr)
    y_true = np.concatenate((y_train_true, y_test_true), axis=0)
    scores = np.concatenate((train_scores, test_scores), axis=0)
    print("Evaluating model with threshold tau=%d" % tau)
    return trainer.evaluate(y_true, scores, threshold=tau)


def store_model(model, export_path: str):
    f = os.open("%s/%s.pt" % (export_path, model.print_name()), "w+")
    model.save(obj=model.state_dict(), f=f)

def train_param(args, timeout_param: str):
    base_path = "C:/Users/verdi/Documents/Datasets/IDS2017/CSV/"
    dataset_path = base_path + timeout_param + '/processed/feature_group_5.npz'
    export_path = base_path + timeout_param + '/results'

    ds = resolve_dataset(args.dataset, dataset_path, args.pct)
    model = DeepSVDD(ds.D(), device=device).to(device)
    trainer = DeepSVDDTrainer(model=model, n_epochs=args.epochs)
    train_ldr, test_ldr = ds.loaders(batch_size=args.batch_size)
    tau = args.tau or int(np.ceil((1 - ds.anomaly_ratio) * 100))

    training_start_time = dt.now()
    print("Training %s with shape %s anomaly ratio %1.4f %s times" % (args.model, ds.shape(), ds.anomaly_ratio, args.n_runs))
    P, R, FS, ROC, PR = [], [], [], [], []
    for i in range(args.n_runs):
        print(f"Run {i}/{args.n_runs}")
        metrics = train_once(trainer, train_ldr, test_ldr, tau)
        print(metrics)
        for m, val in zip([P, R, FS, ROC, PR], metrics.values()):
            m.append(val)
        model.reset()
    P, R, FS, ROC, PR = np.array(P), np.array(R), np.array(FS), np.array(ROC), np.array(PR)
    res = defaultdict()
    for vals, key in zip([P, R, FS, ROC, PR], ["Precision", "Recall", "F1-Score", "AUROC", "AUPR"]):
        res[key] = "%2.4f (%2.4f)" % (vals.mean(), vals.std())
    all_params = {
        "N": ds.N, "runs": args.n_runs, "epochs": args.epochs, "tau": tau, "rho": "%1.4f" % ds.anomaly_ratio, **model.get_params()
    }
    store_results(all_params, res, model.print_name(), args.dataset, dataset_path, training_start_time, export_path)
    # store_model(model, export_path)

if __name__ == '__main__':
    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Selected device %s' % device)

    for param in args.timeout_params[0].split(' '):
        print(f"Training model {args.model} on timeout params {param}")
        train_param(args, param)
   

# 120s-5 'Precision': 0.5214370079473604, 'Recall': 0.5030861201879352, 'F1-Score': 0.5120972168351106, 'AUROC': 0.7969074778900092, 'AUPR': 0.5320399885254098
# 60s-5s 'Precision': 0.29750308614480064, 'Recall': 0.2967345980287, 'F1-Score': 0.297118345169439, 'AUROC': 0.636616872864416, 'AUPR': 0.3579091797973381
# 30s-2.5s 'Precision': 0.39635746717930215, 'Recall': 0.38646359228523225, 'F1-Score': 0.39134800665675973, 'AUROC': 0.7065362937454103, 'AUPR': 0.3535867227630274