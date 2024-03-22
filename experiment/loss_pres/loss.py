from functools import reduce
import numpy as np
import pandas as pd
import os
import sys
from argparse import ArgumentParser

import torch

sys.path.append("../..")
sys.path.append("../../GWAE2SAT")

from GWAE2SAT.data import get_max_shape, get_train_val_dataloader, parity_comparison
from GWAE2SAT.generate import generate
from GWAE2SAT.model import get_vgae
from GWAE2SAT.train import train

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", type=str,
                        help="path to CNFs", default="/home/dcrowley/cnfs/cnfs_train")
    parser.add_argument("--dest", dest="dest", type=str,
                        default="loss_results.csv",
                        help="where to save results")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    max_shape = get_max_shape(args.src)

    train_labels, test_labels = train_test_split(args.src, .05)

    train_dl, val_dl = get_train_val_dataloader(args.src, 16, max_shape, .1, train_labels)

    for loss in ["mae", "mse"]:
        for pres in [9, 99]:
            vgae = get_vgae(max_shape, 32, device,
                            128, 256, 2, 3, .4)
            optim = torch.optim.Adam(vgae.parameters(), .003)
            loss_fn = torch.nn.L1Loss() if loss == "mae" else torch.nn.MSELoss()

            model = train(vgae, train_dl, val_dl, 20, 16,
                          optim, loss_fn, pres, 1., "models/", device)

            generate(model, args.src, f"formulas_{loss}_{pres}", 16, device,
                     None, 1, 128, 256, 2, 3, .4, 10, test_labels)

    lookup = {}

    for sat_file in test_labels:
        lookup[sat_file] = {}
        lookup[sat_file]["original"] = parity_comparison(os.path.join(args.src,
                                                                      sat_file))
        for comparison in ["mae_9", "mae_99", "mse_9", "mse_99"]:
            lookup[sat_file][comparison] = scores_for_sat_file(sat_file,
                                                              f"formulas_{comparison}")

    columns = ["Original", "MAE_9", "MAE_99", "MSE_9", "MSE_99"]
    df = pd.DataFrame(columns=columns)
    
    for sat_instance in lookup.keys():
        try:
            flat = [lookup[sat_instance]["original"]] + [lookup[sat_instance]["mae_9"]]
            flat += [lookup[sat_instance]["mae_99"]] + [lookup[sat_instance]["mse_9"]]
            flat += [lookup[sat_instance]["mse_99"]]
            df.loc[sat_instance] = flat
        except:
            continue

    print(df)
    df.to_csv(args.dest)


def scores_for_sat_file(sat_file, src):
    scores = [parity_comparison(os.path.join(src, sat))
              for sat in os.listdir(src)
              if sat.startswith(f"{sat_file[:-4]}-")]
    return tuple(torch.Tensor(scores).mean(dim=0).numpy())
                       
        
def train_test_split(directory, test_size, seed=440):
    np.random.seed(seed)
    files = np.random.permutation(os.listdir(directory))
    cut = int(len(files) * test_size)

    test = files[:cut]
    train = files[cut:]

    return train, test

    
if __name__ == "__main__":
    main()
