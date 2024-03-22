import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():
    scaler = MinMaxScaler()
    
    df = pd.read_csv("full_results.csv", index_col=None)
    w2_df = pd.read_csv("w2_full_results.csv", index_col=None)
    w2_df = w2_df[w2_df["comp_name"] == "w2sat"]
    df = pd.concat([df, w2_df], axis=0)
    norm_df = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-2]),
                           columns=df.columns[:-2])
    norm_df["comp_name"] = df["comp_name"].to_numpy()
    norm_df["cnf_name"] = df["cnf_name"].to_numpy()
    
    l1_with_w2 = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "Small_GWAE2SAT_Allo": [], "Small_GWAE2SAT_Auto": [],
                  "Big_GWAE2SAT_Allo": [], "Big_GWAE2SAT_Auto": []}
    l1_without_w2 = {"EGNN2S": [], "G2SAT": [], "Small_GWAE2SAT_Allo": [], "Small_GWAE2SAT_Auto": [],
                     "Big_GWAE2SAT_Allo": [], "Big_GWAE2SAT_Auto": []}
    l2_with_w2 = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "Small_GWAE2SAT_Allo": [], "Small_GWAE2SAT_Auto": [],
                  "Big_GWAE2SAT_Allo": [], "Big_GWAE2SAT_Auto": []}
    l2_without_w2 = {"EGNN2S": [], "G2SAT": [], "Small_GWAE2SAT_Allo": [], "Small_GWAE2SAT_Auto": [],
                     "Big_GWAE2SAT_Allo": [], "Big_GWAE2SAT_Auto": []}

    comparisons_for_paper_with_W2SAT = ["EGNN2S", "G2SAT", "W2SAT", "Small_GWAE2SAT_Allo"]
    comparisons_for_paper_without_W2SAT = ["EGNN2S", "G2SAT", "Small_GWAE2SAT_Allo"]
    l1_winners_with_w2 = []
    l2_winners_with_w2 = []
    l1_winners_without_w2 = []
    l2_winners_without_w2 = []
    for instance in df["cnf_name"].unique():
        sub_df = df[df["cnf_name"] == instance]
        original = sub_df[sub_df["comp_name"] == "original"].iloc[0, :-2]
        if (sub_df["comp_name"] == "w2sat").any():
            for comp in l1_with_w2.keys():
                try:
                    pair = sub_df[sub_df["comp_name"] == comp.lower()].iloc[0, :-2]                    
                    l1_with_w2[comp] += [manhattan(original, pair)]
                    l2_with_w2[comp] += [euclidean(original, pair)]
                except:
                    pass
            latest_l1 = [l1_with_w2[comp][-1] for comp in comparisons_for_paper_with_W2SAT]
            latest_l2 = [l2_with_w2[comp][-1] for comp in comparisons_for_paper_with_W2SAT]
            l1_winners_with_w2 += [(comparisons_for_paper_with_W2SAT[np.argmin(latest_l1)])]
            l2_winners_with_w2 += [(comparisons_for_paper_with_W2SAT[np.argmin(latest_l2)])]
        else:
            for comp in l1_without_w2.keys():
                try:
                    pair = sub_df[sub_df["comp_name"] == comp.lower()].iloc[0, :-2]
                    l1_without_w2[comp] += [manhattan(original, pair)]
                    l2_without_w2[comp] += [euclidean(original, pair)]
                except:
                    pass
            latest_l1 = [l1_without_w2[comp][-1] for comp in comparisons_for_paper_without_W2SAT]
            latest_l2 = [l2_without_w2[comp][-1] for comp in comparisons_for_paper_without_W2SAT]
            l1_winners_without_w2 += [(comparisons_for_paper_without_W2SAT[np.argmin(latest_l1)])]
            l2_winners_without_w2 += [(comparisons_for_paper_without_W2SAT[np.argmin(latest_l2)])]
            
    print("L1 with W2")
    print(pd.Series(l1_winners_with_w2).value_counts())
    print()
    print("L2 with W2")
    print(pd.Series(l2_winners_with_w2).value_counts())
    print()
    print("L1 without W2")
    print(pd.Series(l1_winners_without_w2).value_counts())
    print()
    print("L2 without W2")
    print(pd.Series(l2_winners_without_w2).value_counts())
    return

    for key in l1_with_w2.keys():
        print(f"{key} : {len(l1_with_w2[key])}")
    return
    manhattan_with_w2 = {}
    euclidean_with_w2 = {}
    for key in l1_with_w2.keys():
        manhattan_with_w2[key] = np.nanmean(np.array(l1_with_w2[key]))
        euclidean_with_w2[key] = np.nanmean(np.array(l2_with_w2[key]))

    manhattan_without_w2 = {}
    euclidean_without_w2 = {}
    for key in l1_without_w2.keys():
        manhattan_without_w2[key] = np.nanmean(np.array(l1_without_w2[key]))
        euclidean_without_w2[key] = np.nanmean(np.array(l2_without_w2[key]))

    # with open("new_report_with_w2.txt", "w") as file:
    #     for metric in [
    #             "manhattan_with_w2", "euclidean_with_w2",
    #             "manhattan_without_w2", "euclidean_without_w2"
    #     ]:
    #         file.write(f"{metric}:\n")
    #         file.write(f"{eval(metric)}\n\n")
        
        
def manhattan(orig, comp):
    return abs(orig - comp).sum()


def euclidean(orig, comp):
    return (((orig - comp)**2).sum())**0.5


if __name__ == "__main__":
    main()
