from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_avg(cell):
    if type(cell) == pd.Series:
        return cell.apply(get_avg)
    if cell is np.nan:
        return cell
    cell = eval(cell)
    return cell[0]


def get_abs_error(col, df, metric):
    return abs(col - df[f"Original {metric}"])


def write_report(df, metric, no_nan,
                 wins_no_nan, wins_whole):
    cols = [col for col in df.columns if "W2SAT" not in col]
    
    with open(f"{metric}_report.txt", "w") as file:
        file.write("No NaN Averages:\n")
        file.write(f"{df.iloc[no_nan].mean(axis=0)}\n\n")

        file.write("Whole Averages:\n")
        file.write(f"{df.loc[:, cols].mean(axis=0)}\n\n")

        file.write("No NaN Wins:\n")
        file.write(f"{wins_no_nan}\n\n")

        file.write("Whole Wins:\n")
        file.write(f"{wins_whole}\n")
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src",
                        default="/home/dcrowley/experiment/mod_clust/results.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.src)
    for metric in ["clu. VIG", "clu. LIG", "mod. VIG",
                   "mod. LIG", "mod. VCG", "mod. LCG"]:
        cols = [x for x in df.columns if metric in x and "Original" not in x]
        sub_df = df.loc[:, cols]
        sub_df = sub_df.apply(get_avg)
        sub_df = sub_df.apply(get_abs_error, args=(df, metric))
        
        wins_whole = {"EGNN2S": 0, "G2SAT": 0, "Allo-GWAE2SAT": 0, "Auto-GWAE2SAT": 0}
        wins_no_nan = {"EGNN2S": 0, "G2SAT": 0, "W2SAT": 0, "Allo-GWAE2SAT": 0, "Auto-GWAE2SAT": 0}
        nan = []
        for row in range(len(sub_df)):
            best = sub_df.iloc[row].argmin()
            best = cols[best][:-(len(metric) + 1)]
            if sub_df.iloc[row].isna().any():
                nan += [row]
                wins_whole[best] += 1
            else:
                wins_no_nan[best] += 1
                
        for key in wins_whole.keys():
            wins_whole[key] += wins_no_nan[key]

        no_nan = [row for row in range(len(df)) if row not in nan]
        write_report(sub_df, metric, no_nan, wins_no_nan, wins_whole)

        markers = ['o', 'v', '1', 's', '*']
        colours = ['b', 'g', 'r', 'm', 'c']
        fig = plt.subplot()
        fig.tick_params(
            axis='x',
            which="both",
            bottom=False,
            top=False,
            labelbottom=False)
        plt.xticks(ticks=list(range(len(sub_df))))
        fig.grid(visible=True, axis='x', alpha=.6)
        for idx, key in enumerate(wins_whole.keys()):
            fig.scatter(range(len(sub_df)), sub_df[f"{key} {metric}"],
                        c=colours[idx], marker=markers[idx],
                        label=key)
        fig.scatter(no_nan, sub_df.iloc[no_nan][f"W2SAT {metric}"],
                    c=colours[-1], marker=markers[-1],
                    label="W2SAT")
        fig.set(xlabel="Problems")
        fig.set(ylabel="Average Inaccuracy")
        fig.set_ylim(0, 1.0)
        fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{metric}.pdf")
        fig.clear()


if __name__ == "__main__":
    main()
