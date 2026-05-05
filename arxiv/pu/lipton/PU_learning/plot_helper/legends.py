from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(8.4, 1.8))

handles = [
    Line2D([], [], color="steelblue", linewidth=3, label="PU with Test-Time Adaptation"),
    Line2D([], [], color="red",    linewidth=3, label="Supervised Retrain"),
    Line2D([], [], color="darkorange",   linewidth=3, label="Supervised (Plug-In)"),
    Line2D([], [], color="burlywood",  linewidth=3, label="Liang et al"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,
    frameon=True,
    framealpha=0.5,
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
    Line2D([], [], color="burlywood", linestyle="--", linewidth=3, label="Liang et al 2010"),
    Line2D([], [], color="steelblue",  linestyle="-", linewidth=3, label="PU with Test-Time Adaptation"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=3,
    frameon=True,
    framealpha=0.5,
    handlelength=3.0,
    columnspacing=1.6,
    handletextpad=0.8,
    fontsize=16,
)

plt.savefig("temporal_legend.pdf", bbox_inches="tight")
plt.close()


# from matplotlib import pyplot as plt
# from matplotlib.lines import Line2D

# fig = plt.figure(figsize=(9, 2.5))

# handles = [
#     Line2D([], [], color="steelblue",  linestyle="-",  linewidth=3, label="PU Retrain"),
#     # Line2D([], [], color="red",        linestyle="-",  linewidth=3, label="Supervised Retrain"),
#     Line2D([], [], color="red",        linestyle="--", linewidth=3, label="Supervised 2010"),
#     # Line2D([], [], color="darkorange", linestyle="-",  linewidth=3, label="Supervised (Plug-In)"),
#     # Line2D([], [], color="burlywood",  linestyle="-",  linewidth=3, label="Liang et al"),
#     Line2D([], [], color="burlywood",  linestyle="--", linewidth=3, label="Liang et al 2010"),
# ]

# fig.legend(
#     handles=handles,
#     loc="center",
#     ncol=3,
#     frameon=True,
#     framealpha=0.5,
#     handlelength=1.5,
#     columnspacing=1.0,
#     handletextpad=0.6,
#     fontsize=16,
# )

# plt.savefig("unified_legend.pdf", bbox_inches="tight")
# plt.close()
