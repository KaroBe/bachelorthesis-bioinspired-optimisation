# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

from pathlib import Path

import csv
import numpy as np

# ------------------------------------------------------------------------------
problems = ["eil51", "eil76", "kroA100", "gr202"]

eil51file = "quality_development_abc_eil51.csv"
eil76file = "quality_development_abc_eil76.csv"
kroA100file = "quality_development_abc_kroA100.csv"
gr202file = "quality_development_abc_gr202.csv"
files = [eil51file, eil76file, kroA100file, gr202file]

optimum_eil51 = 426
optimum_ei76 = 538
optimum_kroA100 = 21282
optimum_gr202 = 40160
optima = [optimum_eil51, optimum_ei76, optimum_kroA100, optimum_gr202]

iterations_abc = 2002
iterations_aco = 2001
iterations_fa = 2002
iterations = [iterations_abc, iterations_aco, iterations_fa]

# ------------------------------------------------------------------------------

# Einlesen

all_data = []
for problem in problems:
    data_arrays = []
    with open(f"quality_development_fa_{problem}.csv") as dest_f:
        reader = csv.reader(dest_f, delimiter=",")
        #first row, create array of columns
        first_row = next(reader)
        for value in first_row:
            data_arrays.append([])
        for i, data in enumerate(first_row):
            data_arrays[i].append(int(data))
        #append all other values to columns
        for row in reader:
            for i, data in enumerate(row):
                data_arrays[i].append(int(data))
    data_arrays = np.asarray(data_arrays)
    all_data.append(data_arrays)
data = all_data

# Ausgabe

# make figure, axes
plt.style.use("ggplot")
plt.tight_layout()

gs_kw = dict(width_ratios=[1, 1], height_ratios=[1, 1])
fig, axes = plt.subplots(figsize=(5.9, 4.4), ncols=2, nrows=2, gridspec_kw=gs_kw)

all_axes = fig.get_axes()

# Titles
for i, ax in enumerate(all_axes):
    ax.set(title=problems[i], xlabel="iteration", ylabel="quality")
# ax1.set_aspect('equal', 'box')

for i, ax in enumerate(all_axes):
    max_values = []
    for j in range(1, 11):
        max_values.append(data[i][j][0])
    ymax = max(max_values)
    ymin = optima[i]
    margin = (ymax - ymin) * 0.5
    ax.set(xlim=(-0.5, iterations_fa + 0.5), ylim=(ymin - margin, ymax + margin))
    ax.plot(data[i][0], data[i][1])
    ax.axhline(y=optima[i], color="yellowgreen", linewidth=2, linestyle="dashed")
    ax.legend()

# Lables

for ax in all_axes:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        shadow=True,
        ncol=2,
    )

fig.tight_layout()

# Saving to specific directory and file

Path("graphs").mkdir(parents=True, exist_ok=True)

plt.savefig("graphs/fa_convergence.png", format="png")
plt.savefig("graphs/fa_convergence.pgf", format="pgf")

plt.close(fig)
