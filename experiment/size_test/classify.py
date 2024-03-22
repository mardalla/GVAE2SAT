import numpy as np
import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import torch

sys.path.append("../../GWAE2SAT")

from data import get_train_val_dataloader
from model import get_vgae
from preprocessing import cnf_to_adj, pad_smaller_instance
from train import train

def classify(X_train_names, y_train_names,
             X_test_names, y_test_names,
             max_shape, directory, device, model=None):
    X_train, y_train = names_to_numpy(X_train_names, y_train_names,
                                      max_shape, directory, model)
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    X_test, y_test = names_to_numpy(X_test_names, y_test_names,
                                    max_shape, directory, model)    
    return f1_score(y_test, random_forest.predict(X_test))
    

def train_test_labels_split(directory, test_size, seed=440):
    np.random.seed(seed)
    files = np.random.permutation(os.listdir(directory))
    labels_sat, labels_unsat = [], []
    for f in files:
        if f.startswith("sat"):
            labels_sat += [f]
        else:
            labels_unsat += [f]
    cut_sat = int(len(labels_sat) * test_size)
    cut_unsat = int(len(labels_unsat) * test_size)

    X_test = labels_sat[:cut_sat] + labels_unsat[:cut_unsat]
    y_test = cut_sat*[1] + cut_unsat*[0]

    X_train = labels_sat[cut_sat:] + labels_unsat[cut_unsat:]
    y_train = (len(labels_sat)-cut_sat)*[1] + (len(labels_unsat)-cut_unsat)*[0]

    return X_train, y_train, X_test, y_test


def names_to_numpy(X_names, y_names, max_shape,
                   directory, model=None, seed=440):
    np.random.seed(seed)
    length_names = len(X_names)
    indices = np.random.permutation(range(length_names))
    y = []
    for idx, index in enumerate(indices):
        path = os.path.join(directory, X_names[index])
        inst = pad_smaller_instance(cnf_to_adj(path), max_shape)
        if model is None:
            inst = inst.numpy().flatten()
            print(f"{idx}/{length_names}")
        else:
            model.to("cpu")
            with torch.no_grad():
                model.eval()
                inst = model.encode(inst).numpy().flatten()
        try:
            X = np.vstack((X, inst))
        except:
            X = inst
        y += [y_names[index]]
    return X, y


def train_then_classify(X_train_names, y_train_names,
                        X_test_names, y_test_names,
                        latent_size, max_shape, device,
                        enc_layer_size, dec_layer_size,
                        directory):
    models_dir = [model for model in os.listdir("models")
                 if f"z{latent_size}_" in model]
    if (os.path.isdir("models") == False
        or len(models_dir) == 0):
        vgae = get_vgae(max_shape, latent_size, device,
                        enc_layer_size, dec_layer_size,
                        2, 3, .4)
        train_dl, val_dl = get_train_val_dataloader(directory, 16, max_shape,
                                                    .1, X_train_names)
        optim = torch.optim.Adam(vgae.parameters(), lr=.003, weight_decay=0)
        mae_loss = torch.nn.L1Loss()
        location = train(vgae, train_dl, val_dl, 20, 16,
                         optim, mae_loss, 9, 1, "models", device)
    else:
        location = os.path.join("models", models_dir[0])
    vgae = get_vgae(max_shape, latent_size, "cpu",
                    enc_layer_size, dec_layer_size,
                    2, 3, .4)
    vgae.load_state_dict(torch.load(location))
    return classify(X_train_names, y_train_names,
                    X_test_names, y_test_names,
                    max_shape, directory, device, vgae)
