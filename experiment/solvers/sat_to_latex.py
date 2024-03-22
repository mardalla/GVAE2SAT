import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                     index_col="Instance")
    df.iloc[:, 1:] *= 10
    caption = "Satisfiability of Generated Instances"
            
    with open("sat_latex.txt", "w") as file:
        file.write("\\begin{table}\n  \centering\n  ")
        file.write("\caption{")
        file.write(caption)
        file.write("}\n  \label{tab:satis_full}\n  ")
        file.write("\\bigskip\n  \\resizebox{\\textwidth}{!}{\n    ")
        file.write("\\begin{tabular}{|l|c|c|c|c|c|c|}\n      ")
        file.write("\hline\n      ")
        file.write("\\textbf{Instance} & \\textbf{Original} & ")
        file.write("\\textbf{EGNN2S} & \\textbf{G2SAT} & \\textbf{W2SAT} & ")
        file.write("\\textbf{Allo-GWAE2SAT} & \\textbf{Auto-GWAE2SAT} \\\\\n      \\hline\n")

    for row in range(len(df)):
        iloc_row = list(df.iloc[row])
        orig = "SAT" if iloc_row[0] else "UNSAT"
        iloc_row = [f"{int(x)}" if x > -1 else f"{x}"
                    for x in iloc_row[1:]]
        iloc_row = [orig] + iloc_row
            
        with open(f"sat_latex.txt", "a") as file:
            file.write("      ")
            name = df.iloc[row].name
            file.write("\_".join(name.split('_')))
            file.write(" & ")
            file.write(" & ".join(iloc_row))
            file.write(" \\\\\n")
    with open(f"sat_latex.txt", "a") as file:
        file.write("      \hline\n    \end{tabular}\n  }\n\end{table}\n")


if __name__ == "__main__":
    main()
