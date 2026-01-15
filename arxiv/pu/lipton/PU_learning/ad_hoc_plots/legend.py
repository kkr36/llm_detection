from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(6, 1))

ax.plot([], [], "-",  color="black", label="PU 2010")
ax.plot([], [], ":",  color="black", label="PN 2010")
ax.plot([], [], "-",  color="red",   label="PU retrain")
ax.plot([], [], ":",  color="red",   label="PN retrain")
ax.plot([], [], "-",  color="blue",  label="PU no_drop")
ax.plot([], [], ":",  color="blue",  label="PN no_drop")

ax.legend(
    loc="center",
    ncol=6,
    frameon=False,
    columnspacing=1.5,
    handlelength=2.5
)

ax.axis("off")
plt.savefig("shared_legend.pdf", bbox_inches="tight")
plt.close()