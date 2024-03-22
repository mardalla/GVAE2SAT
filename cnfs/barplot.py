import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Assume 'tests' is a list of test instance names and 'lines' is a list of timeout instances
tests = os.listdir("/home/dcrowley/cnfs/cnfs_test")
with open("/home/dcrowley/w2sat/w2sat/time_outs.txt", "r") as file:
    lines = file.readlines()

# Read the CSV file into a DataFrame
df = pd.read_csv("/home/dcrowley/after_experiment/solvers/satisfiability_results_Cadical153_Glucose42_Lingeling_Maplesat.csv",
                 index_col="Instance")["Original"]
lines = [line[:-11] for line in lines]

# Define the check function to update the instances count
def check(line, instances, family_group):
    soluble = df.loc[line]
    family = line.split('_')[1]
    # Group families if specified
    for group, members in family_group.items():
        if family in members:
            family = group
            break
    if family not in instances.keys():
        instances[family] = [0, 0]
    instances[family][int(not soluble)] += 1

# Family groupings
family_group = {
    'clique': ['3clique', '4clique', '5clique'],
    'color': ['3color', '4color', '5color']
}

# Initialize dictionaries to count instances
w2sat_instances = {}
non_w2sat_instances = {}

# Process the test instances
for test in tests:
    check(test, w2sat_instances if test not in lines else non_w2sat_instances, family_group)

# Define the families for plotting (after grouping)
families = ['clique', 'color', 'cliquecoloring', 'dominating', 'matching', 'op', 'php', 'subsetcard', 'tiling', 'tseitin']

# Prepare the data for plotting
w2sat_data = np.array([w2sat_instances.get(family, [0, 0]) for family in families])
non_w2sat_data = np.array([non_w2sat_instances.get(family, [0, 0]) for family in families])

# Plotting
x = np.arange(len(families))  # the label locations
width = 0.35  # the width of the bars
gap = 0.05  # the width of the gap between groups

# Calculate the x locations for the groups
x = np.arange(len(families))  # the label locations

fig, ax = plt.subplots(figsize=(15, 10))

# Plot the bars with a gap between 'w2sat' and 'non w2sat'
ax.bar(x - (width / 2 + gap / 2), w2sat_data[:, 0], width, label='127 Instances SAT', color='blue')
ax.bar(x - (width / 2 + gap / 2), w2sat_data[:, 1], width, bottom=w2sat_data[:, 0], label='127 Instances UNSAT', color='orange')

ax.bar(x + (width / 2 + gap / 2), non_w2sat_data[:, 0], width, label='51 Instances SAT', color='green')
ax.bar(x + (width / 2 + gap / 2), non_w2sat_data[:, 1], width, bottom=non_w2sat_data[:, 0], label='51 Instances UNSAT', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Counts')
# ax.set_title('Counts by instance family and solubility')
ax.set_xticks(x)
ax.set_xticklabels(families, size=12)
ax.set_yticklabels([0, 20, 40, 60, 80, 100, 120, 140], size=12)
ax.legend()

# Rotate the tick labels for better readability
plt.xticks(rotation=45)

# Save the plot as PNG and EPS without numbers above bars
plt.savefig("grouped_bar_chart.png")
plt.savefig("grouped_bar_chart.eps")

# Close the plot to avoid displaying it in an interactive session
plt.close()
