import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("ranking_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                     index_col="Instance")
    caption = "Rankings of Solver Times"

    title = "rank_latex.txt"
    with open(title, "w") as file:
        file.write("\\begin{table}\n  \centering\n  ")
        file.write("\caption{")
        file.write(caption)
        file.write("}\n  \label{tab:rank_full}\n  ")
        file.write("\\bigskip\n  \\resizebox{\\textwidth}{!}{\n    ")
        file.write("\\begin{tabular}{|l|c|c|c|c|c|c|}\n      ")
        file.write("\hline\n      ")
        file.write("\\thead{Instance} & \\thead{Original} & ")
        file.write("\\thead{EGNN2S} & \\thead{G2SAT} & \\thead{W2SAT} & ")
        file.write("\\thead{Allo-GWAE2SAT} & \\thead{Auto-GWAE2SAT} \\\\\n      \\hline\n")

    for row in range(len(df)):
        iloc_row = list(df.iloc[row])
        orig = "".join(iloc_row[0].split()).strip("[]")
        iloc_row = ["".join(x.strip("[]").split(", ")) if type(x) == str else f"{x}"
                    for x in iloc_row[1:]]
        iloc_row = [orig] + iloc_row
            
        with open(title, "a") as file:
            file.write("      ")
            name = df.iloc[row].name
            file.write("\_".join(name.split('_')))
            file.write(" & ")
            file.write(" & ".join(iloc_row))
            file.write(" \\\\\n")
    with open(title, "a") as file:
        file.write("      \hline\n    \end{tabular}\n  }\n\end{table}\n")


if __name__ == "__main__":
    main()
