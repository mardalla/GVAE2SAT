import matplotlib.pyplot as plt
import os
import pandas as pd

tests = os.listdir("/home/dcrowley/cnfs/cnfs_test")

with open("/home/dcrowley/w2sat/w2sat/time_outs.txt", "r") as file:
    lines = file.readlines()

df = pd.read_csv("/home/dcrowley/after_experiment/solvers/satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                 index_col="Instance")["Original"]

lines = [line[:-11] for line in lines]

def check(line, instances):
    soluble = df.loc[line]
    first = line.find('_') + 1
    second = line.find('_', 6)
    family = line[first:second]
    if family not in instances.keys():
        if soluble:
            instances[family] = [1, 0]
        else:
            instances[family] = [0, 1]
    else:
        if soluble:
            instances[family][0] += 1
        else:
            instances[family][1] += 1

w2sat_instances = {}
non_w2sat_instances = {}

for test in tests:
    if test in lines:
        check(test, non_w2sat_instances)
    else:
        check(test, w2sat_instances)

x = []
labels = []
for family in w2sat_instances.keys():
    maximum = max(w2sat_instances[family])
    this_x = [-maximum, -w2sat_instances[family][1], -w2sat_instances[family][1],
              0, 0, w2sat_instances[family][0], w2sat_instances[family][0], maximum]
    x += [this_x]
    labels += [family]
fig = plt.boxplot(x, whis=(25, 75), labels=labels, sym="")
plt.xticks(rotation=45)
plt.ylabel("Number of Satisfiable and Unsatisfiable Instances")
plt.autoscale()
plt.savefig("fig.pdf")
