from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(4, 1))

handles = [
    Line2D([], [], color="blue", linewidth=2, label="PU Retrain"),
    Line2D([], [], color="red",  linewidth=2, label="PN Retrain"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,
    frameon=True,      # matches your screenshot
    handlelength=2.5,
    columnspacing=1.5
)

plt.savefig("alpha_legend.pdf", bbox_inches="tight")
plt.close()

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(4.5, 1.2))

handles = [
    Line2D([], [], color="blue", linestyle=":", linewidth=2, label="PU 2010"),
    Line2D([], [], color="red",  linestyle=":", linewidth=2, label="PN 2010"),
    Line2D([], [], color="blue", linestyle="-", linewidth=2, label="PU Retrain"),
    Line2D([], [], color="red",  linestyle="-", linewidth=2, label="PN Retrain"),
]

fig.legend(
    handles=handles,
    loc="center",
    ncol=2,
    frameon=True,
    handlelength=2.8,
    columnspacing=1.5,
)

plt.savefig("temporal_legend.pdf", bbox_inches="tight")
plt.close()
