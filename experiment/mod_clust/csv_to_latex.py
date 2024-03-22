import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("results.csv", index_col="Instance")
    for metric in ["clu. VIG", "clu. LIG", "mod. VIG", "mod. LIG",
                   "mod. VCG", "mod. LCG"]:
        cols = [col for col in df.columns if metric in col]
        sub_df = df.loc[:, cols]
        caption = metric.split()
        graph = caption[1]
        if caption[0] == "clu.":
            caption = f"{graph} - Average Clustering Coefficient"
        else:
            caption = f"{graph} - Modularity"
            
        with open(f"{metric}_latex.txt", "w") as file:
            file.write("\\begin{table}\n  \centering\n  ")
            file.write("\caption{")
            file.write(caption)
            file.write("}\n  \label{tab:")
            file.write('_'.join(caption.split()))
            file.write("_full}\n  ")
            file.write("\\bigskip\n  \\resizebox{\\textwidth}{!}{\n    ")
            file.write("\\begin{tabular}{|l|c|c|c|c|c|c|}\n      ")
            file.write("\hline\n      ")
            file.write("\\textbf{Instance} & \\textbf{Original} & ")
            file.write("\\textbf{EGNN2S} & \\textbf{G2SAT} & \\textbf{W2SAT} & ")
            file.write("\\textbf{Allo-GWAE2SAT} & \\textbf{Auto-GWAE2SAT} \\\\\n      \\hline\n")
        for row in range(len(sub_df)):
            iloc_row = list(sub_df.iloc[row])
            orig = f"{iloc_row[0]:.2f}"
            iloc_row = [eval(x) if x is not np.nan else x
                        for x in iloc_row[1:]]
            iloc_row = [f"({x[0]:.2f}, {x[1]:.2f})"
                        if x is not np.nan else f"{x}"
                        for x in iloc_row]
            iloc_row = [orig] + iloc_row
            
            with open(f"{metric}_latex.txt", "a") as file:
                file.write("      ")
                name = sub_df.iloc[row].name
                file.write("\_".join(name.split('_')))
                file.write(" & ")
                file.write(" & ".join(iloc_row))
                file.write(" \\\\\n")
        with open(f"{metric}_latex.txt", "a") as file:
            file.write("      \hline\n    \end{tabular}\n  }\n\end{table}\n")


if __name__ == "__main__":
    main()
