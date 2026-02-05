from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(6, 1.2))

handles = [
    Line2D([], [], color="orange", linewidth=2, label="PU Retrain (BBE)"),
    Line2D([], [], color="red",    linewidth=2, label="PN Retrain (BBE)"),
    Line2D([], [], color="green",  linewidth=2, label="MLE Retrain"),
    Line2D([], [], color="blue",   linewidth=2, label="PN (Plug-In)"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,
    frameon=True,
    handlelength=2.8,
    columnspacing=1.6,
    fontsize=14,
)

plt.savefig("alpha_legend.pdf", bbox_inches="tight")
plt.close()


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(6.5, 1.6))

handles = [
    Line2D([], [], color="red",   linestyle=":", linewidth=2, label="PN 2010 (BBE)"),
    Line2D([], [], color="red",   linestyle="-", linewidth=2, label="PN Retrain (BBE)"),
    Line2D([], [], color="green", linestyle=":", linewidth=2, label="MLE 2010"),
    Line2D([], [], color="green", linestyle="-", linewidth=2, label="MLE Retrain"),
    Line2D([], [], color="blue",  linestyle=":", linewidth=2, label="PN 2010 Rescaled (Plug-In)"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,
    frameon=True,
    handlelength=2.8,
    columnspacing=1.6,
    fontsize=14,
)

plt.savefig("temporal_legend.pdf", bbox_inches="tight")
plt.close()
