import pandas as pd

def main():
    # df_orig  = pd.read_csv("../../experiment/mod_clust/results.csv", index_col="Instance")
    # df_big   = pd.read_csv("big/results.csv", index_col="Instance")
    # df_stoch = pd.read_csv("big/stochastic_results.csv", index_col="Instance")
    # df_small = pd.read_csv("stochastic_results.csv", index_col="Instance")

    # df_big.columns   = [f"Big_{col}" for col in df_big.columns]
    # df_small.columns = [f"Small_stoch_{col}" for col in df_small.columns]

    # stoch = [col for col in df_stoch.columns if "Original" not in col]
    # big   = [col for col in df_big.columns   if "Original" not in col]

    # whole = pd.concat([df_orig, df_big.loc[:, big], df_stoch.loc[:, stoch], df_small], axis=1)

    whole = pd.read_csv("results.csv", index_col="Instance")
    gvae_total = pd.read_csv("whole_gvae_results.csv", index_col="Instance")
    whole = pd.concat([whole, gvae_total], axis=1)

    small_allo = [col for col in whole.columns if "Big" not in col and "Auto" not in col]
    whole = whole.loc[:, small_allo]

    w2_sat = pd.read_csv("w2_results.csv", index_col="Instance")
    gvae_total = pd.read_csv("whole_gvae_results.csv", index_col="Instance")
    whole = pd.concat([whole, w2_sat], axis=1)
    
    whole_with_w2 = whole[~whole["W2SAT clu. VIG"].isna()]
    whole_without_w2 = (whole[whole["W2SAT clu. VIG"].isna()]).loc[:, small_allo]

    cluvig = [col for col in whole.columns if "clu. VIG" in col]
    clulig = [col for col in whole.columns if "clu. LIG" in col]
    modvig = [col for col in whole.columns if "mod. VIG" in col]
    modlig = [col for col in whole.columns if "mod. LIG" in col]
    modvcg = [col for col in whole.columns if "mod. VCG" in col]
    modlcg = [col for col in whole.columns if "mod. LCG" in col]

    with open("gvae_total_w2_report.txt", "w") as file:
        for metric in [cluvig, clulig, modvig, modlig, modvcg, modlcg]:
            sub_df = whole_with_w2.loc[:, metric]
            sub_df = sub_df.applymap(first)

            def diff(x):
                return abs(x - sub_df.iloc[:, 0])

            difference = sub_df.iloc[:, 1:].apply(diff)
            difference = difference.dropna()

            wins     = difference.idxmin(axis=1).value_counts()
            averages = difference.mean(axis=0)

            file.write(f"Considered Columns:\n{metric}\n\n")
            file.write(f"Wins per model:\n{wins}\n\n")
            file.write(f"Average error:\n{averages}\n\n\n\n")

    cluvig = [col for col in cluvig if "W2SAT" not in col]
    clulig = [col for col in clulig if "W2SAT" not in col]
    modvig = [col for col in modvig if "W2SAT" not in col]
    modlig = [col for col in modlig if "W2SAT" not in col]
    modvcg = [col for col in modvcg if "W2SAT" not in col]
    modlcg = [col for col in modlcg if "W2SAT" not in col]
    
    with open("gvae_total_non_w2_report.txt", "w") as file:
        for metric in [cluvig, clulig, modvig, modlig, modvcg, modlcg]:
            sub_df = whole_without_w2.loc[:, metric]
            sub_df = sub_df.applymap(first)

            def diff(x):
                return abs(x - sub_df.iloc[:, 0])

            difference = sub_df.iloc[:, 1:].apply(diff)
            difference = difference.dropna()

            wins     = difference.idxmin(axis=1).value_counts()
            averages = difference.mean(axis=0)

            file.write(f"Considered Columns:\n{metric}\n\n")
            file.write(f"Wins per model:\n{wins}\n\n")
            file.write(f"Average error:\n{averages}\n\n\n\n")


def first(x):
    if type(x) == str:
        x = eval(x)
    if type(x) == tuple:
        x = x[0]
    return x


if __name__ == "__main__":
    main()
