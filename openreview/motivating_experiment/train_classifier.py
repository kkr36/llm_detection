import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def load_data(year, chunked, chunk_size):
    conditional_str = f"_{chunk_size}" if chunked else ''
    flipped_conditional_str = f"chunked_" if chunked else ''
    embeddings = np.load(f"/share/garg/openreview_data/{flipped_conditional_str}all_embeddings_{year}{conditional_str}.npy")
    with open(f"/share/garg/openreview_data/{flipped_conditional_str}raw_reviews_{year}{conditional_str}.pickle", 'rb') as f:
        raw_text = pickle.load(f)
    return embeddings, raw_text

if __name__ == "__main__":
    train_year = 2021
    test_years = [2018, 2019, 2020, 2021]
    chunked = False
    chunk_size = 2

    # load and split train data
    embeddings, raw_text = load_data(train_year, chunked, chunk_size)

    val_frac = .1
    train_idx = int(len(embeddings)*val_frac/2)
    train_embeddings = embeddings[train_idx:-train_idx]
    val_embeddings = np.vstack([embeddings[:train_idx], embeddings[-train_idx:]])
    train_text, val_text = raw_text[train_idx:-train_idx], raw_text[:train_idx] + raw_text[-train_idx:]
    train_labels = [0 if i < len(train_embeddings)//2 else 1 for i in range(len(train_embeddings))]
    val_labels = [0 if i < len(val_embeddings)//2 else 1 for i in range(len(val_embeddings))]

    # Define XGBoost classifier
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    # Train
    model.fit(train_embeddings, train_labels)

    # Predict
    y_pred = model.predict(val_embeddings)

    # Evaluate
    for test_year in test_years:
        if test_year == train_year:
            test_embeddings, test_labels, test_text = val_embeddings, val_labels, val_text
        else:
            test_embeddings, test_text = load_data(test_year, chunked, chunk_size)
            test_labels = [0 if i < len(test_embeddings)//2 else 1 for i in range(len(test_embeddings))]
        
        y_pred = model.predict(test_embeddings)
        print(f"Accuracy {test_year}:", accuracy_score(test_labels, y_pred))
        print(classification_report(test_labels, y_pred))

    for test_year in test_years:
        if test_year == train_year:
            test_embeddings, test_labels, test_text = val_embeddings, val_labels, val_text
        else:
            test_embeddings, test_text = load_data(test_year, chunked, chunk_size)
            test_labels = [0 if i < len(test_embeddings)//2 else 1 for i in range(len(test_embeddings))]
        test_embeddings = test_embeddings[len(test_embeddings)//2:,:]
        test_labels = [1 for _ in range(len(test_embeddings))]

        y_pred = model.predict(test_embeddings)
        print(f"Accuracy {test_year}:", accuracy_score(test_labels, y_pred))
        # print(classification_report(test_labels, y_pred))

