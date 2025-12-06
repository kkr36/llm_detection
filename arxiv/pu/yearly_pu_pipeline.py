from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
# from sklearn.random_projection import GaussianRandomProjection

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from tqdm import tqdm
import os
from collections import defaultdict
import pickle

compression = PCA

def plot_cal_curves(Y, Y_hat, filepath):
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(Y, Y_hat, strategy='uniform', n_bins=10)

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, format="pdf")
    plt.clf()

def get_centroids(X, Y):
    mask_human = (Y == -1)
    mask_ai = (Y == 1)
    cent_human = X[mask_human].mean(axis=0) if np.any(mask_human) else np.array([np.nan, np.nan])
    cent_ai = X[mask_ai].mean(axis=0) if np.any(mask_ai) else np.array([np.nan, np.nan])
    return np.vstack([cent_human, cent_ai])

def plot_centroids_over_time(centroids_list, years_axis, title, outpath):
    if len(centroids_list) < 2:
        return  # not enough data to connect segments

    human_pts = np.array([c[0] for c in centroids_list])
    ai_pts = np.array([c[1] for c in centroids_list])

    fig, ax = plt.subplots()
    norm = mcolors.Normalize(vmin=min(years_axis), vmax=max(years_axis))

    for label, pts, marker in [("Human", human_pts, "o"), ("AI", ai_pts, "s")]:
        # Skip plotting if all NaNs
        if np.isnan(pts).all():
            continue
        valid = ~np.isnan(pts[:,0])
        pts = pts[valid]
        yrs = np.array(years_axis)[valid]

        if len(pts) > 1:
            segments = np.array([pts[:-1], pts[1:]]).transpose(1, 0, 2)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(yrs[:-1])
            lc.set_linewidth(3)
            ax.add_collection(lc)
        sc = ax.scatter(pts[:,0], pts[:,1], c=yrs, cmap='viridis', edgecolor='k', s=50, label=label, marker=marker)

    ax.set_xlabel("PCA0")
    ax.set_ylabel("PCA1")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Year")
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close(fig)

def subsample(arr, frac, seed=42):
    # Separate by label
    neg_rows = arr[arr[:, -1] == -1]
    pos_rows = arr[arr[:, -1] == 1]

    # Subsample positive rows
    n_pos = int(len(pos_rows) * frac)
    sub_pos_rows = pos_rows[np.random.choice(len(pos_rows), n_pos, replace=False)]

    # Combine negatives (all) + subsampled positives
    result = np.vstack((neg_rows, sub_pos_rows))

    np.random.seed(seed)
    np.random.shuffle(result)
    return result

if __name__ == "__main__":
    years = list(range(2010,2026))
    centroids_train, centroids_val, centroids_test = [], [], []
    # pcts = []
    # test_pcts = defaultdict(list)

    preds = {}

    for i, year in tqdm(list(enumerate(years))):
        years_axis = years[:i+1]
        output_dir = f"logs/{year}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(f"{output_dir}/test"):
            os.makedirs(f"{output_dir}/test")
        preds[year] = {}

        ### LOAD IN DATA ###
        train_data = np.load(f"/share/garg/arxiv_kaggle/pu/{year}_train.npy")
        train_data = subsample(train_data, 0.5)
        train_X, train_Y = train_data[:,:-1], train_data[:,-1]

        val_data = np.load(f"/share/garg/arxiv_kaggle/pu/{year}_cal.npy")
        val_data = subsample(val_data, 0.5)
        val_X, val_Y = val_data[:,:-1], val_data[:,-1]

        # import pdb; pdb.set_trace()
        # continue


        ### TRANSFORM ###
        n_components = 50  # adjust based on data size and variance explained
        preprocess = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", compression(n_components=n_components, random_state=42)),
        ])
        train_X = preprocess.fit_transform(train_X)
        val_X = preprocess.transform(val_X)

        ### COMPUTE CENTROIDS ###
        # centroids_train.append(get_centroids(train_X, train_Y))
        # centroids_val.append(get_centroids(val_X, val_Y))
        # centroids_test.append(get_centroids(test_X, test_Y))
        # plot_centroids_over_time(centroids_train, years_axis, f"Train Centroids up to {year}", f"logs/centroids_train.pdf")
        # plot_centroids_over_time(centroids_val, years_axis, f"Validation Centroids up to {year}", f"logs/centroids_val.pdf")
        # plot_centroids_over_time(centroids_test, years_axis, f"Test Centroids up to {year}", f"logs/centroids_test.pdf")

        ### PLOT DATA ###
        plt.scatter(train_X[:,0], train_X[:,1], c=train_Y, cmap="viridis", alpha=.2)
        plt.colorbar(label="-1 = human, 1 = LLM")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"{output_dir}/real_train.pdf", format="pdf")
        plt.clf()
    
        plt.scatter(val_X[:,0], val_X[:,1], c=val_Y, cmap="viridis", alpha=.2)
        plt.colorbar(label="-1 = human, 1 = LLM")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"{output_dir}/real_val.pdf", format="pdf")
        plt.clf()

        #### TRAIN ###
        svc = SVC(C=.1, kernel='rbf', gamma=0.01, probability=True, verbose=True)
        pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
        pu_estimator.fit(train_X, train_Y)
        train_probs = pu_estimator.predict_proba(train_X)

        plt.scatter(train_X[:,0], train_X[:,1], c=train_probs[:,0], cmap="viridis", alpha=.2)
        plt.colorbar(label="P(human)")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"{output_dir}/pre_cal_train.pdf", format="pdf")
        plt.clf()

        ### CALIBRATE ###
        val_probs = pu_estimator.predict_proba(val_X)

        plt.scatter(val_X[:,0], val_X[:,1], c=val_probs[:,0], cmap="viridis", alpha=.2)
        plt.colorbar(label="P(human)")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"{output_dir}/pre_cal_val.pdf", format="pdf")
        plt.clf()
        
        platt = CalibratedClassifierCV(pu_estimator, method='sigmoid', cv='prefit')
        platt.fit(val_X, val_Y)  # use validation set for calibration

        calibrated_probs = platt.predict_proba(val_X)
        train_probs = platt.predict_proba(train_X)

        plot_cal_curves(val_Y, calibrated_probs[:,1], f"{output_dir}/cal_curve_val.pdf")

        preds[year]['train'] = ([train_probs[:,1]], train_Y)
        preds[year]['cal'] = (calibrated_probs[:,1], val_Y)

        plt.scatter(val_X[:,0], val_X[:,1], c=calibrated_probs[:,0], cmap="viridis", alpha=.2)
        plt.colorbar(label="P(human)")
        plt.xlabel("PCA0")
        plt.ylabel("PCA1")
        plt.savefig(f"{output_dir}/post_cal_val.pdf", format="pdf")
        plt.clf()

        ### TEST SET EVAL ###
        preds[year]['test'] = {}
        for alpha in [0, 0.05, 0.1, 0.2, 0.3, 0.5]:

            test_data = np.load(f"/share/garg/arxiv_kaggle/pu/{year}_test_{alpha}.npy")
            test_X, test_Y = test_data[:,:-1], test_data[:,-1]
            test_X = preprocess.transform(test_X)

            # plot data
            plt.scatter(test_X[:,0], test_X[:,1], c=test_Y, cmap="viridis", alpha=.2)
            plt.colorbar(label="-1 = human, 1 = LLM")
            plt.xlabel("PCA0")
            plt.ylabel("PCA1")
            plt.savefig(f"{output_dir}/test/real_test_{alpha}.pdf", format="pdf")
            plt.clf()

            test_probs = platt.predict_proba(test_X)
            # test_pcts[alpha].append(np.mean(test_probs[:,1]))
            preds[year]['test'][alpha] = (test_probs[:,1], test_Y)

            plt.scatter(test_X[:,0], test_X[:,1], c=test_probs[:,1], cmap="viridis", alpha=.2)
            plt.colorbar(label="P(AI)")
            plt.xlabel("PCA0")
            plt.ylabel("PCA1")
            plt.savefig(f"{output_dir}/test/test_set_{alpha}.pdf", format="pdf")
            plt.clf()

            plot_cal_curves(test_Y, test_probs[:,1], f"{output_dir}/test/cal_curve_test_{alpha}.pdf")

            ### hist probs
            positives_mask = (test_Y == 1)
            negatives_mask = (test_Y != 1)
            positive_preds = test_probs[positives_mask]
            negative_preds = test_probs[negatives_mask]

            plt.hist(negative_preds[:,1])
            plt.xlabel("pred pct ai")
            plt.savefig(f"{output_dir}/test/test_Human_hist_{alpha}.pdf",format="pdf")
            plt.clf()

            if alpha > 0:
                plt.hist(positive_preds[:,1])
                plt.xlabel("pred pct ai")
                plt.savefig(f"{output_dir}/test/test_AI_hist_{alpha}.pdf",format="pdf")
                plt.clf()

        ### PLOT DS ###
        # plt.plot(years_axis, pcts)
        # plt.xlabel("year")
        # plt.ylabel("pct pred AI")
        # plt.ylim(0,1)
        # plt.savefig("logs/pct_over_time.pdf", format="pdf")
        # plt.clf()

        ### HISTOGRAM PREDS ###
        for x,y,ts in [(train_X, train_Y, "train"), (val_X, val_Y, "val")]:
            pos_mask = (y==1)
            neg_mask = (y==-1)

            for name, mask in [("AI", pos_mask), ("Human", neg_mask)]:
                mask_x = x[mask]
                mask_preds = platt.predict_proba(mask_x)
                plt.hist(mask_preds[:,1])
                plt.xlabel("pct pred ai")
                plt.savefig(f"{output_dir}/{ts}_{name}_hist.pdf", format="pdf")
                plt.clf()

        ### SAVE ###
        save_path = f"/share/garg/arxiv_kaggle/pu/pu_model_{year}_calibrated.pkl"
        joblib.dump({
            "preprocess": preprocess,
            "model": platt
        }, save_path)
        print(f"✅ Calibrated PU model saved to {save_path}")

        with open("/share/garg/arxiv_kaggle/pu/preds.pkl", "wb") as f:
            pickle.dump(preds, f)
