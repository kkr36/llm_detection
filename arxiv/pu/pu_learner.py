from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
import joblib
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

    n_components = 2  # adjust based on data size and variance explained
    preprocess = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=42)),
    ])

    # Fit PCA on training set only
    train_X = preprocess.fit_transform(train_X)
    val_X = preprocess.transform(val_X)

    plt.scatter(val_X[:,0], val_X[:,1], c=val_Y, cmap="viridis", alpha=.2)
    plt.colorbar(label="-1 = human, 1 = LLM")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("real_val.pdf", format="pdf")
    plt.clf()

    # train estimator
    svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True, verbose=True)
    pu_estimator = ElkanotoPuClassifier(estimator=svc, hold_out_ratio=0.2)
    print("fitting")
    pu_estimator.fit(train_X, train_Y)

    # eval on val
    val_probs = pu_estimator.predict_proba(val_X)
    val_preds = pu_estimator.predict(val_X)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        val_Y, val_preds
    )

    plt.scatter(val_X[:,0], val_X[:,1], c=val_probs[:,0], cmap="viridis", alpha=.4)
    plt.colorbar(label="P(human)")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("pre_cal_val.pdf", format="pdf")
    plt.clf()
    
    print("Prior to calibration:")
    print("F1 score: {}".format(f1_score))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    
    # platt scale
    platt = CalibratedClassifierCV(pu_estimator, method='sigmoid', cv='prefit')
    # import pdb; pdb.set_trace()

    platt.fit(val_X, val_Y)  # use validation set for calibration

    calibrated_preds = platt.predict(val_X)
    calibrated_probs = platt.predict_proba(val_X)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        val_Y, calibrated_preds
    )

    plt.scatter(val_X[:,0], val_X[:,1], c=calibrated_probs[:,0], cmap="viridis", alpha=.4)
    plt.colorbar(label="P(human)")
    plt.xlabel("PCA0")
    plt.ylabel("PCA1")
    plt.savefig("post_cal_val.pdf", format="pdf")
    plt.clf()

    print("After calibration:")
    print("F1 score: {}".format(f1_score))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    # save
    save_path = f"/share/garg/arxiv_kaggle/pu/pu_model_{year}_calibrated.pkl"
    joblib.dump({
        "preprocess": preprocess,
        "model": platt
    }, save_path)
    print(f"✅ Calibrated PU model saved to {save_path}")
