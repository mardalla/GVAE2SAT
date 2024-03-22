from functools import reduce
import numpy as np
import os
from numpy.random import permutation
import sys
from torch.utils.data import DataLoader, Dataset

sys.path.append("/home/dcrowley/GWAE2SAT")

from preprocessing import cnf_to_adj, pad_smaller_instance


class SATDataset(Dataset):
    def __init__(self, directory, labels, max_shape, reduced):
        self.labels = labels
        self.directory = directory
        self.max_shape = max_shape
        self.reduced = reduced

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.labels[idx])
        adj = cnf_to_adj(path, self.reduced)
        adj = pad_smaller_instance(adj, self.max_shape)
        return adj


def get_train_val_dataloader(path, batch_size, max_shape, reduced, val_size=0, labels=None):
    if labels is None:
        labels = permutation(os.listdir(path))
    cut = int(len(labels) * val_size)
    training_ds = SATDataset(path, labels[cut:], max_shape, reduced)
    val_ds = SATDataset(path, labels[:cut], max_shape, reduced)
    training_dl = DataLoader(training_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    return training_dl, val_dl


def get_max_shape(path, reduced):    
    max_rows = 0
    max_cols = 0

    for cnf in os.listdir(path):
        adj = cnf_to_adj(os.path.join(path, cnf), reduced)
        rows, cols = adj.size()
        max_rows = rows if (rows > max_rows) else max_rows
        max_cols = cols if (cols > max_cols) else max_cols

    return [max_rows, max_cols]


def get_dataloader_with_names(path, batch_size, max_shape,
                              reduced=True,labels=None):
    if labels is None:
        labels = os.listdir(path)
    dataset = SATDataset(path, labels, max_shape, reduced)
    return DataLoader(dataset, batch_size=batch_size), labels


def parity_comparison(sat_file):
    with open(sat_file, "r") as file:
        clauses = file.readlines()
    while clauses[0][0] == 'p' or clauses[0][0] == 'c':
        clauses = clauses[1:]
    clauses = [clause.split()[:-1] for clause in clauses]
    literals = np.array([int(lit)
                         for lit in reduce(lambda x, y: x+y, clauses)])
    return ((literals > 0).sum(),
            (literals < 0).sum())
