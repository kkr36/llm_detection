import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from collections import defaultdict
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def nested_dict():
    return defaultdict(dict)

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

if __name__ == "__main__":

    metrics = defaultdict(nested_dict)

    train_years = list(range(2010,2026))

    for train_year in tqdm(train_years):
        train_data = np.load(f"/share/garg/arxiv_kaggle/pu/{train_year}_train.npy", allow_pickle=True)
        cal_data = np.load(f"/share/garg/arxiv_kaggle/pu/{train_year}_cal.npy", allow_pickle=True)
        train_data[:, -1][train_data[:, -1] == -1] = 0
        cal_data[:, -1][cal_data[:, -1] == -1] = 0
        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_cal, y_cal = cal_data[:, :-1], cal_data[:, -1]

        # --- Train base XGBoost model ---
        base_model = xgb.XGBClassifier(
            # n_estimators=300,
            # max_depth=5,
            # learning_rate=0.05,
            # subsample=0.8,
            # colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        base_model.fit(X_train, y_train)

        # --- Fit Platt scaling using the calibration set ---
        # method="sigmoid" => Platt scaling (logistic regression)
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv="prefit"   # use the already-fitted base model
        )
        calibrated_model.fit(X_cal, y_cal)

        # --- Evaluate calibration effect ---
        raw_probs = base_model.predict_proba(X_cal)[:, 1]
        calibrated_probs = calibrated_model.predict_proba(X_cal)[:, 1]
        if not os.path.exists(f"cal_curves/{train_year}"):
            os.makedirs(f"cal_curves/{train_year}")
        plot_cal_curves(y_cal, raw_probs, f"cal_curves/{train_year}/base_cal.pdf")
        plot_cal_curves(y_cal, calibrated_probs, f"cal_curves/{train_year}/cal_cal.pdf")

        # histogram of preds on pos/neg in cal set
        pos_cal_ind = np.where(y_cal == 1)
        neg_cal_ind = np.where(y_cal == 0)
        probs_pos = calibrated_probs[pos_cal_ind]
        probs_neg = calibrated_probs[neg_cal_ind]
        std_pos_cal = np.std(probs_pos)
        std_neg_cal = np.std(probs_neg)
        std_cal = np.std(calibrated_probs)
        entropy_pos_cal = entropy(probs_pos, base=2)
        entropy_neg_cal = entropy(probs_neg, base=2)

        plt.hist(probs_pos)
        plt.savefig(f"hists/probs_pos_cal_{train_year}.pdf", format='pdf')
        plt.clf()
        plt.hist(probs_neg)
        plt.savefig(f"hists/probs_neg_cal_{train_year}.pdf", format='pdf')
        plt.clf()
        # import pdb; pdb.set_trace()

        auc_before = roc_auc_score(y_cal, raw_probs)
        auc_after = roc_auc_score(y_cal, calibrated_probs)

        print(f"ROC AUC before calibration: {auc_before:.4f}")
        print(f"ROC AUC after calibration:  {auc_after:.4f}")

        metrics[train_year]['cal']['before_cal'] = auc_before
        metrics[train_year]['cal']['after_cal'] = auc_after
        metrics[train_year]['cal']['neg_std'] = std_neg_cal
        metrics[train_year]['cal']['pos_std'] = std_pos_cal
        metrics[train_year]['cal']['std'] = std_cal
        metrics[train_year]['cal']['neg_entropy'] = entropy_neg_cal
        metrics[train_year]['cal']['pos_entropy'] = entropy_pos_cal

        # test on each test set alpha
        for alpha in [0, 0.05, 0.1, 0.2, 0.3, 0.5]:
            test_alpha = np.load(f"/share/garg/arxiv_kaggle/pu/{train_year}_test_{alpha}.npy")
            test_alpha[:, -1][test_alpha[:, -1] == -1] = 0
            X_test, y_test = test_alpha[:, :-1], test_alpha[:, -1]
            test_probs = calibrated_model.predict_proba(X_test)[:, 1]
            raw_test_probs = base_model.predict_proba(X_test)[:,1]
            plot_cal_curves(y_test, raw_test_probs, f"cal_curves/{train_year}/base_test_{alpha}.pdf")
            plot_cal_curves(y_test, test_probs, f"cal_curves/{train_year}/cal_test_{alpha}.pdf")

            auc_test = accuracy_score(y_test, np.round(test_probs))
            avg_prob = np.mean(test_probs)
            std_prob = np.std(test_probs)

            ### hist probs
            test_probs_pos = test_probs[np.where(y_test == 1)]
            test_probs_neg = test_probs[np.where(y_test == 0)]
            std_prob_pos = np.std(test_probs_pos)
            std_prob_neg = np.std(test_probs_neg)
            test_entropy_pos = entropy(test_probs_pos, base=2)
            test_entropy_neg = entropy(test_probs_neg, base=2)

            if alpha != 0:
                plt.hist(test_probs_pos)
                plt.savefig(f"hists_test/probs_pos_test_{train_year}_{alpha}.pdf", format='pdf')
                plt.clf()

            plt.hist(test_probs_neg)
            plt.savefig(f"hists_test/probs_neg_test_{train_year}_{alpha}.pdf", format='pdf')
            plt.clf()

            metrics[train_year]['test_acc'][alpha] = auc_test
            metrics[train_year]['test_prob'][alpha] = avg_prob
            metrics[train_year]['test_std'][alpha] = std_prob
            metrics[train_year]['test_std_pos'][alpha] = std_prob_pos
            metrics[train_year]['test_std_neg'][alpha] = std_prob_neg
            metrics[train_year]['test_entropy_pos'][alpha] = test_entropy_pos
            metrics[train_year]['test_entropy_neg'][alpha] = test_entropy_neg            
            # import pdb; pdb.set_trace()

    with open("xgb_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)