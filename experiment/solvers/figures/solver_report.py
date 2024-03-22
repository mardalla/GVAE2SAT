from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_avg(cell):
    if type(cell) == pd.Series:
        return cell.apply(get_avg)
    if cell is np.nan:
        return cell
    cell = np.array(eval(cell))
    return cell.mean(), cell.std()


def get_first(cell):
    if type(cell) == pd.Series:
        return cell.apply(get_first)
    return cell[0]


def fix_original(cell):
    return [int(c) for c in cell.strip("[]").split()]


def hamming_distance(order1, order2):
    distance = 0
    for idx, rank in enumerate(order1):
        distance += rank != order2[idx]
    return distance


def pairwise_distance(order1, order2):
    distance = 0
    for left in range(len(order1) - 1):
        for right in range(left, len(order1)):
            lval = order1[left]
            rval = order1[right]
            if order2.index(lval) > order2.index(rval):
                distance += 1
    return distance


def score_ordering(cell, original):
    if cell is np.nan:
        return np.nan, np.nan
    cell = eval(cell)
    return (hamming_distance(original, cell),
            pairwise_distance(original, cell))


def satisfiability_report(df):
    sat_df = df[df["Original"] == True]
    unsat_df = df[df["Original"] == False]

    with open("sat_report.txt", "w") as file:
        pass
    
    for dataframe in ["sat_df", "unsat_df"]:
        name = dataframe
        dataframe = eval(name)
        no_nan = dataframe[~dataframe["W2SAT"].isna()]
        no_nan_avgs = no_nan.loc[
            :,
            ["EGNN2S", "G2SAT", "W2SAT",
             "Allo-GWAE2SAT", "Auto-GWAE2SAT"]].mean(axis=0)

        whole_avgs = dataframe.loc[
            :,
            ["EGNN2S", "G2SAT",
             "Allo-GWAE2SAT", "Auto-GWAE2SAT"]].mean(axis=0)

        with open("sat_report.txt", "a") as file:
            file.write(f"{name} No NaN:\n")
            file.write(f"{no_nan_avgs}\n\n")

            file.write(f"{name} Whole:\n")
            file.write(f"{whole_avgs}\n\n")
        

def times_report(df, suffix):
    df = df.apply(get_avg)
    df.to_csv(f"mean_std_times{suffix}", index_label="Instance")

    no_nan = df[~df["W2SAT"].isna()]
    no_nan_avgs = no_nan.apply(get_first).mean(axis=0)

    not_w2sat = [col for col in df.columns if "W2SAT" not in col]
    whole = df.loc[:, not_w2sat]
    whole_avgs = whole.apply(get_first).mean(axis=0)

    with open("times_report.txt", "w") as file:
        file.write("No NaN:\n")
        file.write(f"{no_nan_avgs}\n\n")

        file.write("Whole:\n")
        file.write(f"{whole_avgs}\n")


def ranking_report(df):
    df["Original"] = df["Original"].apply(fix_original)
    wins_whole = {"EGNN2S": [0, 0], "G2SAT": [0, 0], "Allo-GWAE2SAT": [0, 0], "Auto-GWAE2SAT": [0, 0]}
    wins_no_nan = {"EGNN2S": [0, 0], "G2SAT": [0, 0], "W2SAT": [0, 0],
                   "Allo-GWAE2SAT": [0, 0], "Auto-GWAE2SAT": [0, 0]}
    hamming = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "Allo-GWAE2SAT": [], "Auto-GWAE2SAT": []}
    pairwise = {"EGNN2S": [], "G2SAT": [], "W2SAT": [], "Allo-GWAE2SAT": [], "Auto-GWAE2SAT": []}

    for row in range(len(df)):
        record = df.iloc[row]
        best_hamming = 5
        best_pairwise = 7
        hamming_winners = []
        pairwise_winners = []
        for col in record.index[1:]:
            hamming_score, pairwise_score = score_ordering(record[col], record["Original"])
            hamming[col] += [hamming_score]
            pairwise[col] += [pairwise_score]
            if hamming_score < best_hamming:
                hamming_winners = [col]
                best_hamming = hamming_score
            elif hamming_score == best_hamming:
                hamming_winners += [col]
            if pairwise_score < best_pairwise:
                pairwise_winners = [col]
                best_pairwise = pairwise_score
            elif pairwise_score == best_pairwise:
                pairwise_winners += [col]
            
        if record.isna().any():
            for winner in hamming_winners:
                wins_whole[winner][0] += 1
            for winner in pairwise_winners:
                wins_whole[winner][1] += 1
        else:
            for winner in hamming_winners:
                wins_no_nan[winner][0] += 1
            for winner in pairwise_winners:
                wins_no_nan[winner][1] += 1

    for distance in ["hamming", "pairwise"]:
        draw_ranking_figure(eval(distance), f"{distance}.pdf")
        for key in hamming.keys():
            df[f"{key} {distance}"] = eval(distance)[key]

    for key in wins_whole:
        for idx in range(2):
            wins_whole[key][idx] += wins_no_nan[key][idx]

    df.to_csv("ranking_with_distances.csv", index_label="Instance")

    write_rank_report(hamming, pairwise, wins_whole, wins_no_nan)


def write_rank_report(hamming, pairwise, wins_whole, wins_no_nan):
    with open(f"ranking_report.txt", "w") as file:
        file.write("No NaN Wins:\n")
        file.write(f"{wins_no_nan}\n\n")

        file.write("Whole Wins:\n")
        file.write(f"{wins_whole}\n\n")

    for dist_name in ["hamming", "pairwise"]:
        distance = eval(dist_name)
        distance = pd.DataFrame(distance)
        cols = [col for col in distance.columns if "W2SAT" not in col]

        with open("ranking_report.txt", "a") as file:
            file.write(f"No NaN mean {dist_name}:\n")
            file.write(f"{distance[~distance.isna().any(axis=1)].mean(axis=0)}\n\n")

            file.write(f"Whole mean {dist_name}:\n")
            file.write(f"{distance.loc[:, cols].mean(axis=0)}\n\n")


def draw_ranking_figure(dist_dict, dest):
    markers = ['o', 'v', '1', 's', '*']
    colours = ['b', 'g', 'r', 'm', 'c']
    fig = plt.subplot()
    fig.tick_params(
        axis='x',
        which="both",
        bottom=False,
        top=False,
        labelbottom=False)
    plt.xticks(ticks=list(range(len(dist_dict["EGNN2S"]))))
    fig.grid(visible=True, axis='x', alpha=.6)
    for idx, key in enumerate(dist_dict.keys()):
        fig.scatter(range(len(dist_dict[key])), dist_dict[key],
                    c=colours[idx], marker=markers[idx],
                    label=key)
    no_nan = []
    w2sat = []
    for jdx, x in enumerate(dist_dict["W2SAT"]):
        if x is not np.nan:
            no_nan += [jdx]
            w2sat += [x]
    fig.scatter(no_nan, w2sat,
                c=colours[-1], marker=markers[-1],
                label="W2SAT")
    fig.set(xlabel="Problems")
    fig.set(ylabel="Distance from Original Ranking")
    fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(dest)
    fig.clear()


def main():
    parser = ArgumentParser()
    parser.add_argument("--src", dest="src",
                        default="/home/dcrowley/experiment/solvers/")
    parser.add_argument("--suffix", dest="suffix",
                        default="_results_Cadical153_Glucose42_Lingeling_Maplesat.csv")
    args = parser.parse_args()

    # df = pd.read_csv(f"{args.src}satisfiability{args.suffix}")
    # satisfiability_report(df)

    # df = pd.read_csv(f"{args.src}times{args.suffix}", index_col="Instance")
    # times_report(df, args.suffix)

    df = pd.read_csv(f"{args.src}ranking{args.suffix}", index_col="Instance")
    ranking_report(df)

    
if __name__ == "__main__":
    main()
