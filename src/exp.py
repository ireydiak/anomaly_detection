from collections import defaultdict
from datetime import datetime as dt
from typing import List
import numpy as np
import pandas as pd
import os
from src.datamanager.dataset import *
from src.trainers.DeepSVDDTrainer import DeepSVDDTrainer
from src.trainers.MemAETrainer import MemAETrainer
from src.trainers.AutoEncoder import DAGMMTrainer, NeuTralADTrainer
from src.trainers.EnergyTrainer import DSEBMTrainer
from src.tabular.OneClass import DeepSVDD
from src.tabular.AutoEncoder import MemAE, DAGMM, NeuTralAD
from src.tabular.Energy import DSEBM
import neptune.new as neptune
from dotenv import dotenv_values


def resolve_dataset(dataset_name: str, path_to_dataset: str, pct: float) -> AbstractDataset:
    clsname = globals()[f'{dataset_name}Dataset']
    return clsname(path_to_dataset, pct)


def store_results(results: dict, params: dict, model_name: str, dataset: str, path: str, start_time: dt, output_path: str = None):
    output_dir = output_path if output_path else f'./results/{dataset}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_dir + '/' + f'{model_name}_results.txt', 'a') as f:
        hdr = "Experiments on {}-{}\n".format(
            start_time.strftime("%d/%m/%Y %H:%M:%S"), dt.now().strftime("%d/%m/%Y %H:%M:%S")
        )
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({path}\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")


def train_once(trainer, train_ldr, test_ldr, tau, nep):
    trainer.train(train_ldr, nep)
    y_train_true, train_scores = trainer.test(train_ldr)
    y_test_true, test_scores = trainer.test(test_ldr)
    y_true = np.concatenate((y_train_true, y_test_true), axis=0)
    scores = np.concatenate((train_scores, test_scores), axis=0)
    print("Evaluating model with threshold tau=%d" % tau)
    return trainer.evaluate(y_true, scores, threshold=tau)


def store_model(model, export_path: str):
    f = os.open("%s/%s.pt" % (export_path, model.print_name()), "w+")
    model.save(obj=model.state_dict(), f=f)


def resolve_model_trainer(model_name: str, params: dict, dataset: str):
    model, trainer = None, None
    if model_name == 'DeepSVDD':
        model = DeepSVDD(params['D'], device=params['device'])
        trainer = DeepSVDDTrainer(model=model, n_epochs=params['epochs'])
    elif model_name == 'MemAE':
        model = MemAE(
            D=params['D'], rep_dim=params.get('rep_dim', 1), mem_dim=params.get('mem_dim', 50), device=params['device']
        )
        trainer = MemAETrainer(
            model=model, alpha=params['alpha'], n_epochs=params['epochs'], batch_size=params['batch_size']
        )
    elif model_name == 'DAGMM':
        model = DAGMM(
            D=params['D'], L=params.get('L', 1), K=params.get('K', 4), device=params['device']
        )
        trainer = DAGMMTrainer(
            model=model, device=params['device'], n_epochs=params['epochs'], batch_size=params['batch_size']
        )
    elif model_name == 'DSEBM-e':
        model = DSEBM(
            D=params['D']
        )
        trainer = DSEBMTrainer(
            model=model, D=params['D'], score='e', device=params['device'], n_epochs=params['epochs'],
            batch_size=params['batch_size']
        )
    elif model_name == 'DSEBM-r':
        model = DSEBM(
            D=params['D']
        )
        trainer = DSEBMTrainer(
            model=model, D=params['D'], score='r', device=params['device'], n_epochs=params['epochs'],
            batch_size=params['batch_size']
        )
    elif model_name == 'NeuTralAD':
        model = NeuTralAD(
            D=params['D'],
            N=params['N'],
            dataset=dataset,
            n_layers=params.get('n_layers', 3),
            temperature=params.get('temperature', 1.0),
        )
        trainer = NeuTralADTrainer(
            model=model,
            device=params['device'], n_epochs=params['epochs'], batch_size=params['batch_size']
        )

    return model, trainer


def load_neptune(mode: str, tags: list) -> neptune.Run:
    cfg = dotenv_values()
    return neptune.init(
        project=cfg["neptune_project"],
        api_token=cfg["neptune_api_token"],
        tags=tags,
        mode=mode
    )


class Experiment:
    def __init__(self, path_to_dataset: str, export_path: str, device: str,
                 neptune_mode: str, neptune_tags: list, dataset: str, pct: float, model: str, epochs: int, n_runs: int,
                 batch_size: int, model_param=None, seed=None):
        if model_param is None:
            self.model_param = {}
        else:
            self.model_param = {}
        if os.path.exists(path_to_dataset):
            self.path_to_dataset = path_to_dataset
        else:
            raise Exception(f"given path {path_to_dataset} does not exist")
        self.export_path = export_path
        self.device = device
        self.neptune_mode = neptune_mode
        self.neptune_tags = neptune_tags
        self.dataset = dataset
        self.pct = pct
        self.model = model
        self.epochs = epochs
        self.tau = None
        self.batch_size = batch_size
        self.seed = seed
        self.n_runs = n_runs

    def get_params(self):
        return {
            'model': self.model,
            'device': self.device,
            'epochs': self.epochs,
            'dataset': self.dataset,
            'pct': self.pct,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'n_runs': self.n_runs,
            **self.model_param
        }


class BatchTrainer:

    def __init__(self, experiments: List[Experiment]):
        self.experiments = experiments

    def summary(self):
        experiments = [exp.get_params() for exp in self.experiments]
        return pd.DataFrame(experiments)

    def train(self):
        for exp in self.experiments:
            self.run_experiment(exp)

    def run_experiment(self, exp: Experiment):
        # (neptune) initialize neptune
        run = load_neptune(exp.neptune_mode, exp.neptune_tags)

        # Load train_set, test_set, trainer and model
        ds = resolve_dataset(exp.dataset, exp.path_to_dataset, exp.pct)
        # TODO: load params from exp
        model, trainer = resolve_model_trainer(
            exp.model,
            {'D': ds.D, 'N': ds.N, 'alpha': 2e-4, 'device': exp.device, 'epochs': exp.epochs, 'batch_size': exp.batch_size, 'temperature': 0.07, 'mem_dim': 500},
            exp.dataset
        )
        train_ldr, test_ldr = ds.loaders(batch_size=exp.batch_size, seed=exp.seed)

        # (neptune) log datasets sizes
        run["data/train/size"] = len(train_ldr)
        run["data/test/size"] = len(test_ldr)

        # (neptune) set parameters to be uploaded
        tau = exp.tau or int(np.ceil((1 - ds.anomaly_ratio) * 100))
        all_params = {
            "N": ds.N,
            "runs": exp.n_runs,
            "anomaly_threshold": tau,
            "anomaly_ratio": "%1.4f" % ds.anomaly_ratio,
            **trainer.get_params()
        }
        run["parameters"] = all_params

        training_start_time = dt.now()
        print("Training %s with shape %s, anomaly ratio %1.4f" % (exp.model, ds.shape(), ds.anomaly_ratio))
        P, R, FS, ROC, PR = [], [], [], [], []
        for i in range(exp.n_runs):
            print(f"Run {i + 1}/{exp.n_runs}")
            metrics = train_once(trainer, train_ldr, test_ldr, tau, run)
            print(metrics)
            for m, val in zip([P, R, FS, ROC, PR], metrics.values()):
                m.append(val)
            # (neptune) log metrics for this run
            for nep_key, dict_key in zip(['precision', 'recall', 'fscore'], ['Precision', 'Recall', 'F1-Score']):
                run["training/metrics/run_{}/{}".format(i, nep_key)].log(metrics[dict_key], step=i)
            model.reset()
        P, R, FS, ROC, PR = np.array(P), np.array(R), np.array(FS), np.array(ROC), np.array(PR)
        res = defaultdict()
        for vals, key in zip([P, R, FS, ROC, PR], ["Precision", "Recall", "F1-Score", "AUROC", "AUPR"]):
            res[key] = "%2.4f (%2.4f)" % (vals.mean(), vals.std())
            # (neptune) store final metrics
            run["training/evaluation/" + key] = res[key]
        store_results(
            res, all_params, exp.model, exp.dataset, exp.path_to_dataset, training_start_time, exp.export_path
        )
        # store_model(model, export_path)
        run.stop()
