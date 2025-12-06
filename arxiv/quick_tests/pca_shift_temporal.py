# train pca on real/llm, apply on another real, why would real shift from non-zero center back to zero-center?

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    year = 2010

    # load in pu data
    train_data = np.load(f"/share/garg/arxiv_kaggle/pu/train_{year}.npy")
    train_X, train_Y = train_data[:,:-1], train_data[:,-1]
    train_Y[np.where(train_Y == 0)] = -1

    val_data = np.load(f"/share/garg/arxiv_kaggle/pu/val_{year}_train.npy")
    val_X, val_Y = val_data[:,:-1], val_data[:,-1]
    val_Y[np.where(val_Y == 0)] = -1
    just_human_val = val_X[np.where(val_Y == -1)]

    n_components = 2  # adjust based on data size and variance explained
    preprocess = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])

    # Fit PCA on training set only
    train_X = preprocess.fit_transform(train_X)
    val_X = preprocess.transform(val_X)
    human_val_X = preprocess.transform(just_human_val)

    plt.scatter(train_X[:,0], train_X[:,1], c=train_Y, cmap="viridis", alpha=.2)
    plt.colorbar(label="-1 = human, 1 = LLM")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("all_train.pdf", format="pdf")
    plt.clf()

    plt.scatter(val_X[:,0], val_X[:,1], c=val_Y, cmap="viridis", alpha=.2)
    plt.colorbar(label="-1 = human, 1 = LLM")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("all_val.pdf", format="pdf")
    plt.clf()

    plt.scatter(human_val_X[:,0], human_val_X[:,1], alpha=.2)
    # plt.colorbar(label="-1 = human, 1 = LLM")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("just_human_val.pdf", format="pdf")
    plt.clf()

    val_years = list(range(2010,2026,1))
    centersx, centersy = [], []

    for i, val_year in enumerate(val_years):

        filename = f"/share/garg/arxiv_kaggle/pu/val_{val_year}.npy"
        val_raw = np.load(filename)[:10000]
        val_data = preprocess.transform(val_raw)

        # plot the raw val data
        plt.scatter(val_data[:,0], val_data[:,1], alpha=.3)
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"over_time/{val_year}.pdf", format="pdf")
        plt.clf()
        print(f"({np.mean(val_data[:,0])} , {np.mean(val_data[:,1])})")
        centersx.append(np.mean(val_data[:,0]))
        centersy.append(np.mean(val_data[:,1]))
    
    plt.scatter(centersx, centersy, c=[i+2010 for i in range(len(centersx))], cmap='viridis')
    plt.colorbar(label="Year")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.xlim((-2.1,2.1))
    plt.ylim((-1.3,1.3))
    plt.savefig("over_time/centers.pdf", format="pdf")