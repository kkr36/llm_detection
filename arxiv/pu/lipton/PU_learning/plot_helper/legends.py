from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(8.4, 1.8))

handles = [
    Line2D([], [], color="steelblue", linewidth=3, label="PU Retrain"),
    Line2D([], [], color="red",    linewidth=3, label="Supervised Retrain"),
    Line2D([], [], color="darkorange",   linewidth=3, label="Supervised (Plug-In)"),
    Line2D([], [], color="burlywood",  linewidth=3, label="Liang et al"),
    Line2D([], [], color="black",  linewidth=3, label="Unachievable Optimal"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,                 # 3 columns → 2 rows with 5 entries
    frameon=True,
    handlelength=3.0,
    columnspacing=1.6,
    handletextpad=0.8,
    fontsize=16,
)

plt.savefig("alpha_legend.pdf", bbox_inches="tight")
plt.close()




from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# SAME size as the previous legend
fig = plt.figure(figsize=(8.4, 1.8))

handles = [
    Line2D([], [], color="red",   linestyle="--", linewidth=3, label="Supervised 2010"),
    Line2D([], [], color="black", linestyle="-", linewidth=3, label="Unachievable Optimal"),
    Line2D([], [], color="burlywood", linestyle="--", linewidth=3, label="Liang et al 2010"),
    Line2D([], [], color="darkorange",  linestyle="--", linewidth=3, label="Supervised 2010 Platt (Plug-In)"),
    # Line2D([], [], color="green", linestyle="-", linewidth=3, label="Liang et al Retrain"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,                 # 3 × 2 layout
    frameon=True,
    handlelength=3.0,
    columnspacing=1.6,
    handletextpad=0.8,
    fontsize=16,
)

plt.savefig("temporal_legend.pdf", bbox_inches="tight")
plt.close()
