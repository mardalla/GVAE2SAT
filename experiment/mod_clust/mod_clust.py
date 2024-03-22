from numpy import nan
import os
import pandas as pd
import subprocess
import sys
from argparse import ArgumentParser

import torch

sys.path.append("../..")

from w2sat.w2sat.utils import eval_solution, read_sat
from experiment.run_models import run_i4vk, run_W2SAT, run_GWAE2SAT

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src", type=str, default="/home/dcrowley/cnfs/cnfs_test/",
                        help="path to test instances to regenerate from")
    parser.add_argument("--EGNN2S", dest="EGNN2S", type=str,
                        help="path to EGNN2S", default="/home/dcrowley/i4vk/SAT_generators/EGNN2S")
    parser.add_argument("--G2SAT", dest="G2SAT", type=str,
                        help="path to G2SAT", default="/home/dcrowley/g2sat/SAT_generators/G2SAT")
    parser.add_argument("--W2SAT", dest="W2SAT", type=str, default="/home/dcrowley/w2sat/w2sat",
                        help="path to W2SAT")
    parser.add_argument("--GWAE2SAT", dest="GWAE2SAT", default="/home/dcrowley/GWAE2SAT",
                        help="path to GWAE2SAT")
    parser.add_argument("--gwae2sat", dest="gwae2sat_train", default="/home/dcrowley/cnfs/cnfs_train/",
                        help="the path to training instances if retraining GWAE2SAT")
    args = parser.parse_args()

    egnn2s_formulas = os.path.join(args.EGNN2S, "formulas")
    g2sat_formulas = os.path.join(args.G2SAT, "formulas")
    w2sat_formulas = os.path.join(args.W2SAT, "result/generation")
    gwae2sat_allo = os.path.join(args.GWAE2SAT, "allo_generations")
    gwae2sat_auto = os.path.join(args.GWAE2SAT, "auto_generations")

    try:
        assert len(os.listdir(egnn2s_formulas)) > 0
    except:
        run_i4vk(args.src, args.EGNN2S)
    print("EGNN2S ready.")
    
    try:
        assert len(os.listdir(g2sat_formulas)) > 0
    except:
        run_i4vk(args.src, args.G2SAT)
    print("G2SAT ready.")

    try:
        assert len(os.listdir(w2sat_formulas)) > 0
    except:
        run_W2SAT(args.src)
    print("W2SAT ready.")

    try:
        assert len(os.listdir(gwae2sat_allo)) > 0
    except:
        run_GWAE2SAT(args.src, args.gwae2sat_train, "allo")
    print("Allo-GWAE2SAT ready.")

    try:
        assert len(os.listdir(gwae2sat_auto)) > 0
    except:
        run_GWAE2SAT(args.src, args.src, "auto")
    print("Auto-GWAE2SAT ready.")

    lookup = {}
    
    for sat_file in os.listdir(args.src):
        lookup[sat_file] = {}
        
        lookup[sat_file]["original"] = get_metrics(os.path.join(args.src, sat_file))
        try:
            lookup[sat_file]["egnn2s"] = scores_for_sat_file(sat_file,
                                                             egnn2s_formulas,
                                                             '_')
        except:
            lookup[sat_file]["egnn2s"] = 6 * [nan]
        try:
            lookup[sat_file]["g2sat"] = scores_for_sat_file(sat_file,
                                                            g2sat_formulas,
                                                            '_')
        except:
            lookup[sat_file]["g2sat"] = 6 * [nan]
        try:
            instance = os.path.join(w2sat_formulas, sat_file)
            scores = torch.Tensor([get_metrics(os.path.join(instance, sample))
                                   for sample in os.listdir(instance)])
            lookup[sat_file]["w2sat"] = list(zip(scores.mean(dim=0).numpy(),
                                                 scores.std(dim=0).numpy()))
        except:
            lookup[sat_file]["w2sat"] = 6 * [nan]
        lookup[sat_file]["allo_gwae2sat"] = scores_for_sat_file(sat_file,
                                                           gwae2sat_allo,
                                                           '-')
        lookup[sat_file]["auto_gwae2sat"] = scores_for_sat_file(sat_file,
                                                                gwae2sat_auto,
                                                                '-')

    features = [
            "clu. VIG",
            "clu. LIG",
            "mod. VIG",
            "mod. LIG",
            "mod. VCG",
            "mod. LCG"
            ]
    comparisons = ["Original", "EGNN2S", "G2SAT", "W2SAT",
                   "Allo-GWAE2SAT", "Auto-GWAE2SAT"]
    columns = [f"{comp} {feat}" for comp in comparisons for feat in features]
    df = pd.DataFrame(columns=columns)

    for sat_instance in lookup.keys():
        try:
            flat = lookup[sat_instance]["original"]
            flat += lookup[sat_instance]["egnn2s"]+lookup[sat_instance]["g2sat"]
            flat += lookup[sat_instance]["w2sat"]+lookup[sat_instance]["allo_gwae2sat"]
            flat += lookup[sat_instance]["auto_gwae2sat"]
            df.loc[sat_instance] = flat
        except:
            continue

    print(df)
    df.to_csv("results.csv", index_label="Instance")
                                      

def get_metrics(path):
    num_vars, _, sat_instance = read_sat(path)
    return eval_solution(sat_instance, num_vars)


def scores_for_sat_file(sat_file, directory, joining_char):
    samples = [cnf for cnf in os.listdir(directory)
               if cnf.startswith(f"{sat_file[:-4]}{joining_char}")]
    scores = torch.Tensor([get_metrics(os.path.join(directory, sample))
                           for sample in samples])
    return list(zip(scores.mean(dim=0).numpy(),
                    scores.std(dim=0).numpy()))
        
    
if __name__ == "__main__":
    main()
