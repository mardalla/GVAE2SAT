import pandas as pd

df = pd.read_csv("satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                 index_col="Instance")
gvae = pd.read_csv("gvae_total_satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                   index_col="Instance")
df = pd.concat([df, gvae], axis=1)
w2_df = pd.read_csv("w2_satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                    index_col="Instance")

with_w2 = w2_df[~w2_df["W2SAT"].isna()].index
without_w2 = w2_df[w2_df["W2SAT"].isna()].index

whole = pd.concat([df, w2_df], axis=1)

df_with_w2 = whole.loc[with_w2, :]
df_without_w2 = whole.loc[without_w2, df.columns]

sat_with = df_with_w2[df_with_w2["Original"]==True]
sat_without = df_without_w2[df_without_w2["Original"]==True]

unsat_with = df_with_w2[df_with_w2["Original"]==False]
unsat_without = df_without_w2[df_without_w2["Original"]==False]

print("SAT with")
print(sat_with.mean(axis=0))
print()

print("SAT without")
print(sat_without.mean(axis=0))
print()

print("UNSAT with")
print(unsat_with.mean(axis=0))
print()

print("UNSAT without")
print(unsat_without.mean(axis=0))
print()
