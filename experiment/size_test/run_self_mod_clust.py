import numpy as np
import pandas as pd
import os
import sys
import time
from argparse import ArgumentParser

import torch

sys.path.append("../..")
sys.path.append("../../GWAE2SAT")

from experiment.mod_clust.mod_clust import get_metrics, scores_for_sat_file
from GWAE2SAT.data import get_max_shape, get_train_val_dataloader
from GWAE2SAT.generate import generate
from GWAE2SAT.model import get_vgae
from GWAE2SAT.train import train
from w2sat.w2sat.utils import eval_solution, read_sat

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", type=str,
                        help="path to CNFs", default="/home/dcrowley/cnfs/cnfs_train")
    parser.add_argument("--dest", dest="dest", type=str,
                        default="size_results.csv",
                        help="where to save results")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    max_shape = get_max_shape(args.src)

    train_labels, test_labels = train_test_split(args.src, .05)

    train_dl, val_dl = get_train_val_dataloader(args.src, 16, max_shape, .1, train_labels)

    vgae32 = get_vgae(max_shape, 32, device,
                      128, 256, 2, 3, .4)
    optim = torch.optim.Adam(vgae32.parameters(), .003)
    mae_loss = torch.nn.L1Loss()

    time0 = time.time()
    model32 = train(vgae32, train_dl, val_dl, 20, 16,
                    optim, mae_loss, 9, 1., "models/", device)
    time_train32 = time.time() - time0

    time0 = time.time()
    generate(model32, args.src, "formulas32", 16, device, None, 1,
             128, 256, 2, 3, .4, 10, test_labels)
    time_gen32 = time.time() - time0

    vgae1024 = get_vgae(max_shape, 1024, device,
                        1024, 1024, 2, 3, .4)
    optim = torch.optim.Adam(vgae1024.parameters(), .003)

    time0 = time.time()
    model1024 = train(vgae1024, train_dl, val_dl, 20, 16,
                      optim, mae_loss, 9, 1., "models/", device)
    time_train1024 = time.time() - time0

    time0 = time.time()
    generate(model1024, args.src, "formulas1024", 16, device, None, 1,
             1024, 1024, 2, 3, .4, 10, test_labels)
    time_gen1024 = time.time() - time0

    with open("time_log.txt", "w") as file:
        file.writelines([
            f"Train 32: {time_train32}\n",
            f"Gen 32: {time_gen32}\n",
            f"Train 1024: {time_train1024}\n",
            f"Gen 1024: {time_gen1024}\n"])

    lookup = {}

    for sat_file in test_labels:
        lookup[sat_file] = {}
        lookup[sat_file]["original"] = get_metrics(os.path.join(args.src, sat_file))
        lookup[sat_file]["32"] = scores_for_sat_file(sat_file,
                                                     "formulas32",
                                                     '-')
        lookup[sat_file]["1024"] = scores_for_sat_file(sat_file,
                                                      "formulas1024",
                                                      '-')

    features = [
            "clu. VIG",
            "clu. LIG",
            "mod. VIG",
            "mod. LIG",
            "mod. VCG",
            "mod. LCG"
            ]
    comparisons = ["Original", "32", "1024"]
    columns = [f"{comp} {feat}" for comp in comparisons for feat in features]
    df = pd.DataFrame(columns=columns)

    for sat_instance in lookup.keys():
        try:
            flat = lookup[sat_instance]["original"] + lookup[sat_instance]["32"]
            flat += lookup[sat_instance]["1024"]
            df.loc[sat_instance] = flat
        except:
            continue

    print(df)
    df.to_csv(args.dest, index_label="Instance")

    
def train_test_split(directory, test_size, seed=440):
    np.random.seed(seed)
    files = np.random.permutation(os.listdir(directory))
    cut = int(len(files) * test_size)

    test = files[:cut]
    train = files[cut:]

    return train, test

    
if __name__ == "__main__":
    main()
