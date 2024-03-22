import numpy as np
import os
import pandas as pd
import subprocess

from GWAE2SAT.preprocessing import cnf_to_adj

"""
  sat_subsetcard_60_2820_64659.cnf
  sat_5color_250_1050_86180.cnf
  sat_subsetcard_60_2820_64659
"""

def main():
    dictfict = {}

    path = "cnfs/cnfs_train"
    for cnf in os.listdir(path):
        adj = cnf_to_adj(os.path.join(path, cnf), False)
        square = adj.size(0)
        adj = cnf_to_adj(os.path.join(path, cnf))
        dictfict[cnf] = [square, adj.size(0), adj.size(1)]

    indices = ["Square", "Clauses", "Vars"]
    df = pd.DataFrame(dictfict, index=indices)
    miss_max = [col for col in df.columns if (col != "sat_subsetcard_60_2820_64659.cnf"
                                              and col != "sat_5color_250_1050_86180.cnf")]
    df = df.loc[:, miss_max].T

    for idx in indices:
        df = randomly_choose_big(df, idx)

    choice = np.random.choice(df.index, 60, False)
    for cnf in choice:
        subprocess.call(["mv", os.path.join(path, cnf), "cnfs/cnfs_test"])
        

def randomly_choose_big(df, by, number=20):
    top_100 = df.sort_values(by=[by], ascending=False).iloc[:100].index
    choice = np.random.choice(top_100, number, False)
    for cnf in choice:
        subprocess.call(["mv", os.path.join("cnfs/cnfs_train", cnf), "cnfs/cnfs_test"])

    unchosen = [row for row in df.index if row not in choice]
    return df.loc[unchosen, :]
    
    
def check_max(path):
    max_clauses = [0, ""]
    max_vars = [0, ""]
    max_combined = [0, ""]
    
    for cnf in os.listdir(path):
        adj = cnf_to_adj(os.path.join(path, cnf))
        if adj.size(0) > max_clauses[0]:
            max_clauses = [adj.size(0), cnf]
        if adj.size(1) > max_vars[0]:
            max_vars = [adj.size(1), cnf]
        adj = cnf_to_adj(os.path.join(path, cnf), False)
        if adj.size(0) > max_combined[0]:
            max_combined = [adj.size(0), cnf]
            
    print(f"Clauses: {max_clauses}")
    print(f"Vars: {max_vars}")
    print(f"Square: {max_combined}")


if __name__ == "__main__":
    main()
