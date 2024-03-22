from functools import reduce
import numpy as np
import pandas as pd
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.append("../..")
sys.path.append("../../GWAE2SAT")

from experiment.mod_clust.mod_clust import get_metrics, scores_for_sat_file
from GWAE2SAT.data import get_max_shape, get_train_val_dataloader, parity_comparison
from GWAE2SAT.generate import generate
from GWAE2SAT.model import get_vgae
from GWAE2SAT.train import train
from w2sat.w2sat.utils import eval_solution, read_sat

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", type=str,
                        help="path to CNFs", default="/home/dcrowley/cnfs/cnfs_train")
    parser.add_argument("--dest", dest="dest", type=str,
                        default="beta_results.csv",
                        help="where to save results")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    max_shape = get_max_shape(args.src)

    train_labels, test_labels = train_test_split(args.src, .05)

    train_dl, val_dl = get_train_val_dataloader(args.src, 16, max_shape, .1, train_labels)

    comparisons = []
    for batch in [1, 4, 8, 16]:
        for pres in [1, 4, 9, 36]:
            for forcing in [True, False]:
                comp = f"batch{batch}_pres{pres}_forcing{forcing}"
                comparisons += [comp]
                vgae = get_vgae(max_shape, 32, device,
                                128, 256, 2, 3, .4)
                optim = torch.optim.Adam(vgae.parameters(), .06, weight_decay=.1)
                mae_loss = torch.nn.L1Loss()

                model = train(vgae, train_dl, val_dl, 8, batch,
                              optim, mae_loss, pres, .001, "models/", device)

                generate(model, args.src,
                         f"formulas_{comp}", batch,
                         device, "clause", 1, 128, 256, 2, 3, .4, 10, test_labels, forcing)

    lookup = {}

    for sat_file in test_labels:
        lookup[sat_file] = {}
        original = os.path.join(args.src, sat_file)
        lookup[sat_file]["original"] = get_metrics(original)
        lookup[sat_file]["original"] += [parity_comparison(original)]
        for comp in comparisons:
            lookup[sat_file][comp] = scores_for_sat_file(sat_file,
                                                         f"formulas_{comp}")

    features = [
            "clu. VIG",
            "clu. LIG",
            "mod. VIG",
            "mod. LIG",
            "mod. VCG",
            "mod. LCG",
            "parity"
            ]
    comparisons = ["original"] + comparisons
    columns = [f"{comp} {feat}" for comp in comparisons for feat in features]
    df = pd.DataFrame(columns=columns)

    for sat_instance in lookup.keys():
        try:
            flat = reduce(lambda x, y: x+y, [lookup[sat_instance][comp]
                                             for comp in comparisons])
            df.loc[sat_instance] = flat
        except:
            continue

    print(df)
    df.to_csv(args.dest, index_label="Instance")


def scores_for_sat_file(sat_file, directory):
    samples = [cnf for cnf in os.listdir(directory)
               if cnf.startswith(f"{sat_file[:-4]}-")]
    structural = torch.Tensor([get_metrics(os.path.join(directory,
                                                        sample))
                               for sample in samples])
    structural = list(zip(structural.mean(dim=0).numpy(),
                          structural.std(dim=0).numpy()))
    literals = [parity_comparison(os.path.join(directory, sat))
                for sat in samples]
    literals = tuple(torch.Tensor(literals).mean(dim=0).numpy())
    return structural + [literals]
                       
    
def train_test_split(directory, test_size, seed=440):
    np.random.seed(seed)
    files = np.random.permutation(os.listdir(directory))
    cut = int(len(files) * test_size)

    test = files[:cut]
    train = files[cut:]

    return train, test

    
if __name__ == "__main__":
    main()
