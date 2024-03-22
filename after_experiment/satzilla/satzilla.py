from argparse import ArgumentParser
import os
import pandas as pd
import subprocess
import sys

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sys.path.append("../../SfP/SATfeatPy")
sys.path.append("..")
sys.path.append("../..")

from run_models import run_i4vk, run_W2SAT, run_GWAE2SAT
from sat_instance.sat_instance import SATInstance

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

    egnn2s_formulas = os.path.join(args.EGNN2S, "fixed_formulas")
    g2sat_formulas = os.path.join(args.G2SAT, "fixed_formulas")
    w2sat_formulas = os.path.join(args.W2SAT, "result/generation")
    small_gwae2sat_allo = os.path.join(args.GWAE2SAT, "small_allo_generations")
    small_gwae2sat_auto = os.path.join(args.GWAE2SAT, "small_auto_generations")
    big_gwae2sat_allo = os.path.join(args.GWAE2SAT, "big_allo_generations")
    big_gwae2sat_auto = os.path.join(args.GWAE2SAT, "big_auto_generations")
    
    # try:
    #     assert len(os.listdir(egnn2s_formulas)) > 0
    # except:
    #     run_i4vk(args.src, args.EGNN2S)
    # print("EGNN2S ready.")
    
    # try:
    #     assert len(os.listdir(g2sat_formulas)) > 0
    # except:
    #     run_i4vk(args.src, args.G2SAT)
    # print("G2SAT ready.")

    # try:
    #     assert len(os.listdir(w2sat_formulas)) > 0
    # except:
    #     run_W2SAT(args.src)
    # print("W2SAT ready.")

    for cnf in os.listdir(args.src):
        try:
            path = os.path.join(args.src, cnf)
            original = get_metrics(path)
            original["comp_name"] = "original"
            original["cnf_name"] = cnf
            original = pd.DataFrame([original])
            try:
                df = pd.concat([df, original])
            except:
                df = original
            
            # try:
            #     egnn2s = get_average(egnn2s_formulas, cnf, "egnn2s", '_')
            #     df = pd.concat([df, egnn2s])
            # except:
            #     pass

            # try:
            #     g2sat = get_average(g2sat_formulas, cnf, "g2sat", '_')
            #     df = pd.concat([df, g2sat])
            # except:
            #     pass

            try:
                w2sat = get_average(w2sat_formulas, cnf, "w2sat", w2sat=True)
                df = pd.concat([df, w2sat])
            except:
                pass

            # gwae2sat = get_average(small_gwae2sat_allo, cnf, "small_gwae2sat_allo", '-')
            # df = pd.concat([df, gwae2sat])
            # gwae2sat = get_average(small_gwae2sat_auto, cnf, "small_gwae2sat_auto", '-')
            # df = pd.concat([df, gwae2sat])
            # gwae2sat = get_average(big_gwae2sat_allo, cnf, "big_gwae2sat_allo", '-')
            # df = pd.concat([df, gwae2sat])
            # gwae2sat = get_average(big_gwae2sat_auto, cnf, "big_gwae2sat_auto", '-')
            # df = pd.concat([df, gwae2sat])
        except:
            continue
    df.to_csv("w2_full_results.csv", index=False)

    df = pd.read_csv("w2_full_results.csv", index_col=None)

    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(df.iloc[:, :-2])
    tsne_result_df = pd.DataFrame({"tsne_1": tsne_result[:, 0],
                                   "tsne_2": tsne_result[:, 1],
                                   "comp_name" : df["comp_name"],
                                   "cnf_name":  df["cnf_name"]})
    tsne_result_df.to_csv("w2_tsne_results.csv", index=False)

    
    # fig, ax = plt.subplots(1)
    # sns.scatterplot(x="tsne_1", y="tsne_2", hue="label",
    #                 data=tsne_result_df, ax=ax)
    # lim = (tsne_result.min()-5, tsne_result.max()+5)
    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.set_aspect("equal")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # plt.savefig("tsne.pdf")

    
def get_metrics(path):
    sat_instance = SATInstance(path, preprocess=False)
    sat_instance.gen_basic_features()
    record = sat_instance.features_dict
    return record


def get_average(directory, sat_file, comp,
                join_char=' ', w2sat=False):
    if w2sat:
        samples = [os.path.join(sat_file, samp)
                   for samp in os.listdir(os.path.join(directory,
                                                       sat_file))]
    else:
        samples = [cnf for cnf in os.listdir(directory)
                   if cnf.startswith(f"{sat_file[:-4]}{join_char}")]
    scores = 0
    for samp in samples:
        try:
            metrics = get_metrics(os.path.join(directory,
                                               samp))
        except:
            continue
        try:
            scores = pd.concat([scores,
                                pd.DataFrame([metrics])])
        except:
            scores = pd.DataFrame([metrics])
    averages = scores.mean()
    averages["comp_name"] = comp
    averages["cnf_name"] = sat_file
    return pd.DataFrame([averages])

                                 
if __name__ == "__main__":
    main()
