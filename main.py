import argparse
import torch
from src.datamanager.dataset import *
from src.trainers.DeepSVDDTrainer import DeepSVDDTrainer
from src.models.OneClass import DeepSVDD
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='The selected model', choices=['DeepSVDD'], type=str)
parser.add_argument('-d', '--dataset', help='The selected dataset', choices=['KDD10', 'NSLKDD', 'USBIDS'], type=str)
parser.add_argument('-p', '--dataset-path', help='Path to the selected dataset', type=str)
parser.add_argument('-b', '--batch-size', help='Size of mini-batch', default=128, type=int)
parser.add_argument('-e', '--epochs', help='Number of training epochs', default=100, type=int)
parser.add_argument('--pct', help='Percentage of original data to keep', default=1., type=float)


def resolve_dataset(dataset_name: str, path_to_dataset: str, pct: float) -> AbstractDataset:
    clsname = globals()[f'{dataset_name}Dataset']
    return clsname(path_to_dataset, pct)


if __name__ == '__main__':
    args = parser.parse_args()
    logger = logging.getLogger()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Selected device %s' % device)

    ds = resolve_dataset(args.dataset, args.dataset_path, args.pct)
    model = DeepSVDD(ds.D(), device=device).to(device)
    trainer = DeepSVDDTrainer(model=model, n_epochs=args.epochs)
    train_ldr, test_ldr = ds.loaders(batch_size=args.batch_size)
    print("Training %s with shape %s" % (args.model, ds.shape()))
    trainer.train(train_ldr)
    metrics = trainer.test(test_ldr)
    print(metrics)
