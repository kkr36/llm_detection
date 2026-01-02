import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_semeval_split(split, split_dir='/home/ubuntu/data/Task_A'):
    data = pd.read_parquet(f"{split_dir}/{split}.parquet")
    
    # compute lengths
    data["length"] = data["code"].str.len()
    data = data[data["length"] < 20000]

    # global min/max
    Lmin = data["length"].min()
    Lmax = data["length"].max()

    # build shared bins (change n_bins if you want)
    n_bins = 40
    bins = np.linspace(Lmin, Lmax, n_bins + 1)

    plt.figure(figsize=(8,5))
    if split == "test":
        plt.hist(
            data["length"],
            bins=bins,
            alpha=0.4,
        )
    else:
        for label, subset in data.groupby("label"):
            plt.hist(
                subset["length"],
                bins=bins,
                alpha=0.4,
                label=f"label {label}"
            )
        plt.legend()
    # plt.xscale("log")

    plt.title(f"Length Distribution ({split})")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"length_dist_{split}.pdf", format="pdf")
    plt.clf()

