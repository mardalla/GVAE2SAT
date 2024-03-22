from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import sys

import torch
from pysat.formula import CNF
from pysat.solvers import Cadical153, Glucose42, Lingeling, Maplesat

sys.path.append("../..")

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
        print(sat_file)
        lookup[sat_file] = {}
        
        lookup[sat_file]["original"] = get_metrics(os.path.join(args.src, sat_file))
        print("     original")
        try:
            lookup[sat_file]["egnn2s"] = scores_for_sat_file(sat_file,
                                                             egnn2s_formulas,
                                                             '_')
            print("     egnn2s")
        except:
            lookup[sat_file]["egnn2s"] = {}
            for metric in ["satisfiability", "times", "ranking"]:
                lookup[sat_file]["egnn2s"][metric] = np.nan
        try:
            lookup[sat_file]["g2sat"] = scores_for_sat_file(sat_file,
                                                            g2sat_formulas,
                                                            '_')
            print("     g2sat")
        except:
            lookup[sat_file]["g2sat"] = {}
            for metric in ["satisfiability", "times", "ranking"]:
                lookup[sat_file]["g2sat"][metric] = np.nan
        try:
            lookup[sat_file]["w2sat"] = scores_for_sat_file(sat_file,
                                                            w2sat_formulas,
                                                            w2sat=True)
            print("     w2sat")
        except:
            lookup[sat_file]["w2sat"] = {}
            for metric in ["satisfiability", "times", "ranking"]:
                lookup[sat_file]["w2sat"][metric] = np.nan
        lookup[sat_file]["allo-gwae2sat"] = scores_for_sat_file(sat_file,
                                                                gwae2sat_allo,
                                                                '-')
        print("     allo-gwae2sat")
        lookup[sat_file]["auto-gwae2sat"] = scores_for_sat_file(sat_file,
                                                                gwae2sat_auto,
                                                                '-')
        print("     auto-gwae2sat")

    save_dataframe("satisfiability", lookup)
    save_dataframe("times", lookup)
    save_dataframe("ranking", lookup)


def scores_for_sat_file(sat_file, directory, joining_char=' ', w2sat=False):
    if w2sat:
        samples = [os.path.join(sat_file, samp)
                   for samp in os.listdir(os.path.join(directory,
                                                       sat_file))]
    else:
        samples = [cnf for cnf in os.listdir(directory)
                   if cnf.startswith(f"{sat_file[:-4]}{joining_char}")]
    scores = pd.DataFrame([get_metrics(os.path.join(directory, sample))
                           for sample in samples])
    all_ten = {}
    all_ten["satisfiability"] = scores["satisfiability"].mean()
    all_ten["times"] = torch.Tensor(scores["times"]).mean(dim=0).numpy()
    all_ten["ranking"] = list(all_ten["times"].argsort())
    all_ten["times"] = list(all_ten["times"])
    return all_ten


def get_metrics(path, solvers=[Cadical153, Glucose42, Lingeling, Maplesat]):
    cnf = CNF(from_file=path)

    results = {}
    avgs = []
    for Solver in solvers:
        times = []
        for t in range(10):
            solver = Solver(bootstrap_with=cnf.clauses, use_timer=True)
            results["satisfiability"] = solver.solve()
            times += [solver.time()]
            print(f"          {Solver}")
            solver.delete
        times = np.array(times)
        avgs += [times.mean()]
    results["times"] = avgs
    results["ranking"] = np.array(avgs).argsort()
    return results
    

def save_dataframe(metric, lookup,
                   columns=["Original", "EGNN2S", "G2SAT", "W2SAT",
                            "Allo-GWAE2SAT", "Auto-GWAE2SAT"],
                   solvers=[Cadical153, Glucose42, Lingeling, Maplesat]):
    df = pd.DataFrame(columns=columns)

    for sat_instance in lookup.keys():
        try:
            flat = [lookup[sat_instance][col.lower()][metric] for col in columns]
            df.loc[sat_instance] = flat
        except:
            continue

    df.to_csv(f"{metric}_results_"
              + "_".join([f"{s}"[22:-2] for s in solvers])
              + ".csv", index_label="Instance")
    
    
if __name__ == "__main__":
    main()
