
import numpy as np
import joblib
from matplotlib import pyplot as plt

if __name__ == "__main__":

    year = 2010
    model_path = f"/share/garg/arxiv_kaggle/pu/pu_model_{year}_calibrated.pkl"
    bundle = joblib.load(model_path)
    preprocess = bundle["preprocess"]
    platt_model = bundle["model"].estimator

    val_years = list(range(2010,2026,1)) 
    # val_years = [2010, 2011, 2012, 2013, 2014, 2016, 2018, 2020, 2022, 2024]
    pcts = []

    for i, val_year in enumerate(val_years):

        filename = f"/share/garg/arxiv_kaggle/pu/val_{val_year}.npy"
        val_raw = np.load(filename)
        val_data = preprocess.transform(val_raw)

        test_probs = platt_model.predict_proba(val_data)[:,1]
        test_labels = np.round(test_probs)
        pcts.append(np.mean(test_labels))
        print(f"{val_year} | {np.mean(test_labels)}")

        # plot the raw val data
        plt.scatter(val_data[:,0], val_data[:,1], c=test_probs, cmap="viridis")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"over_time/{val_year}.pdf")
        plt.clf()

    plt.plot(val_years, pcts)
    plt.xlabel("year")
    plt.ylabel("pct predicted AI")
    plt.savefig("over_time/overtime.pdf")
    plt.clf()