from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import sys
import torch

sys.path.append("/home/dcrowley/GWAE2SAT/")

from data import get_train_val_dataloader
from model import get_vgae
from preprocessing import cnf_to_adj, pad_smaller_instance
from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", dest="model",
                        default="/home/dcrowley/GWAE2SAT/models/"
                                + "max_shape[2820, 250]_ep200_bat16_lr0.03"
                                + "_wd0.1_z32_pres9_beta0.001.pt")
    parser.add_argument("--src", dest="src", default="xyz")
    parser.add_argument("--dest", dest="dest", default="random_classification_results.csv")
    parser.add_argument("--test", dest="test", default=.1)
    parser.add_argument("--seed", dest="seed", default=200)
    parser.add_argument("--latent", dest="latent", default=32)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    max_shape = args.model[args.model.find("max_shape")+9:]
    max_shape = max_shape[:max_shape.find(']')+1]
    max_shape = eval(max_shape)

    (X_train_names, y_train_names,
     X_test_names, y_test_names) = [], [], [], []
    for prefix in set([x[:5] for x in os.listdir(args.src)]):
        problems = [prob for prob in os.listdir(args.src)
                    if prob.startswith(prefix)]
        problems = list(np.random.permutation(problems))
        cut = int(len(problems) * args.test)
        X_test = problems[:cut]
        X_train = problems[cut:]
        y_test = [0 if name[1] == 'u' else 1 for name in X_test]
        y_train = [0 if name[1] == 'u' else 1 for name in X_train]

        X_train_names += X_train
        X_test_names += X_test
        y_train_names += y_train
        y_test_names += y_test

    dictionary = {}
    dictionary["full"] = [classify(X_train_names, y_train_names,
                                   X_test_names, y_test_names,
                                   max_shape, args.src, device)]
    print(f"Full: {dictionary['full']}")
    pd.DataFrame(dictionary).to_csv(args.dest, index=False)
    
    vgae = get_vgae(max_shape, args.latent, "cpu",
                    128, 256,
                    2, 3, .4)
    vgae.load_state_dict(torch.load(args.model))

    dictionary["structured_model"] = [classify(X_train_names, y_train_names,
                                               X_test_names, y_test_names,
                                               max_shape, args.src, device, vgae)]
    print(f"Model trained on structured data: {dictionary['structured_model']}")
    pd.DataFrame(dictionary).to_csv(args.dest, index=False)

    dictionary["random_model"] = [train_then_classify(X_train_names, y_train_names,
                                                      X_test_names, y_test_names,
                                                      16, max_shape, device,
                                                      128, 256, args.src)]
    print(f"Model trained on random data: {dictionary['random_model']}")
    pd.DataFrame(dictionary).to_csv(args.dest, index=False)

    
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
    restart = True
    if os.path.isdir("models"):
        models_dir = [model for model in os.listdir("models")
                      if f"z{latent_size}_" in model]
        if len(model_dir) > 0:
            restart = False
            
    if restart:
        vgae = get_vgae(max_shape, latent_size, device,
                        enc_layer_size, dec_layer_size,
                        2, 3, .4)
        train_dl, val_dl = get_train_val_dataloader(directory, 16, max_shape,
                                                    .1, X_train_names)
        optim = torch.optim.Adam(vgae.parameters(), lr=.03, weight_decay=.1)
        mae_loss = torch.nn.L1Loss()
        location = train(vgae, train_dl, val_dl, 200, 16,
                         optim, mae_loss, 9, .001, "models", device)
    else:
        location = os.path.join("models", models_dir[0])
    vgae = get_vgae(max_shape, latent_size, "cpu",
                    enc_layer_size, dec_layer_size,
                    2, 3, .4)
    vgae.load_state_dict(torch.load(location))
    return classify(X_train_names, y_train_names,
                    X_test_names, y_test_names,
                    max_shape, directory, device, vgae)


if __name__ == "__main__":
    main()
