import pandas as pd


def main():
    df = pd.read_csv("tsne_results.csv", index_col=None)
    title = "tsne_latex.txt"
    caption = "t-SNE of SATzilla Features"
    
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

    for instance in df["cnf_name"].unique():
        sub_df = df[df["cnf_name"] == instance]
        with open(title, "a") as file:
            cnf = "\_".join(instance.split('_'))
            file.write(f"      {cnf} ")
            for column in ["original", "egnn2s", "g2sat", "w2sat",
                           "gwae2sat_allo", "gwae2sat_auto"]:
                cell = sub_df[sub_df["comp_name"] == column]
                try:
                    file.write(f"& ({cell.iloc[0, 0]:.2f}, {cell.iloc[0, 1]:.2f}) ")
                except:
                    file.write("& nan ")
            file.write("\\\\\n")
    with open(title, "a") as file:
        file.write("      \hline\n    \end{tabular}\n  }\n\end{table}\n")
        

if __name__ == "__main__":
    main()
