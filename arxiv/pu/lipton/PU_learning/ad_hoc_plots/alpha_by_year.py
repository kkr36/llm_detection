alphas = [.7722222222222221, .6906862745098039, .5914673561732385, .46142322097378274, .5057537399309552, .20013431833445267]
years = [2014, 2016, 2018, 2020, 2023, 2025]

font = {
        # 'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20
    }
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rc('font', **font)
plt.figure(figsize=(12,9))
ax = plt.gca()
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.plot(years, alphas)
plt.ylabel("Alpha")
plt.xlabel("Unlabeled Year")
plt.xticks(years)
plt.tight_layout()
plt.savefig("alpha_year.pdf", bbox_inches="tight")