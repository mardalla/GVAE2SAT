import numpy as np
import pandas as pd

def main():
    df = pd.read_csv("tsne_results.csv", index_col=None)

    l1_no_nan = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "GWAE2SAT_Allo": [], "GWAE2SAT_Auto": []}
    l1_whole = {"EGNN2S": [], "G2SAT": [], "GWAE2SAT_Allo": [], "GWAE2SAT_Auto": []}
    l2_no_nan = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "GWAE2SAT_Allo": [], "GWAE2SAT_Auto": []}
    l2_whole = {"EGNN2S": [], "G2SAT": [], "GWAE2SAT_Allo": [], "GWAE2SAT_Auto": []}
    
    for instance in df["cnf_name"].unique():
        sub_df = df[df["cnf_name"] == instance]
        original = sub_df[sub_df["comp_name"] == "original"].iloc[0, :2]
        try:
            w2sat = sub_df[sub_df["comp_name"] == "w2sat"].iloc[0, :2]
            for comp in l1_no_nan.keys():
                pair = sub_df[sub_df["comp_name"] == comp.lower()].iloc[0, :2]
                l1_no_nan[comp] += [manhattan(original, pair)]
                l2_no_nan[comp] += [euclidean(original, pair)]
        except:
            for comp in l1_whole.keys():
                pair = sub_df[sub_df["comp_name"] == comp.lower()].iloc[0, :2]
                l1_whole[comp] += [manhattan(original, pair)]
                l2_whole[comp] += [euclidean(original, pair)]

    manhattan_no_nan = {}
    euclidean_no_nan = {}
    for key in l1_no_nan.keys():
        manhattan_no_nan[key] = np.array(l1_no_nan[key]).mean()
        euclidean_no_nan[key] = np.array(l2_no_nan[key]).mean()

    manhattan_whole = {}
    euclidean_whole = {}
    for key in l1_whole.keys():
        manhattan_whole[key] = np.array(l1_whole[key] + l1_no_nan[key]).mean()
        euclidean_whole[key] = np.array(l2_whole[key] + l2_no_nan[key]).mean()

    with open("tsne_report.txt", "w") as file:
        for metric in ["manhattan_no_nan", "euclidean_no_nan",
                       "manhattan_whole", "euclidean_whole"]:
            file.write(f"{metric}:\n")
            file.write(f"{eval(metric)}\n\n")
        
        
def manhattan(orig, comp):
    return abs(orig - comp).sum()


def euclidean(orig, comp):
    return (((orig - comp)**2).sum())**0.5


if __name__ == "__main__":
    main()
