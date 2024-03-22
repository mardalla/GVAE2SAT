import numpy as np
import pandas as pd


def to_list(x):
    if type(x) == str:
        x = eval(x)
    return np.mean(x)

    
def main():
    df = pd.read_csv("times_results_Cadical153_Glucose42_Lingeling_Maplesat.csv", index_col="Instance")
    w2_df = pd.read_csv("w2_times_results_Cadical153_Glucose42_Lingeling_Maplesat.csv", index_col="Instance")
    df = pd.concat([df, w2_df], axis=1)
    with_w2 = w2_df[~w2_df["W2SAT"].isna()].index
    without_w2 = w2_df[w2_df["W2SAT"].isna()].index
    
    for col in df.columns:
        df[col] = df[col].apply(to_list)

    print("With W2")
    print(df.loc[with_w2, :].mean(axis=0))
    print()

    print("Without W2")
    print(df.loc[without_w2, :].mean(axis=0))

if __name__ == "__main__":
    main()
