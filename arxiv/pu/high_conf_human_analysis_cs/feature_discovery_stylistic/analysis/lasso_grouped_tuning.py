#!/usr/bin/env python3
"""Train-only grouped-CV tuning for L1 logistic year prediction.

This script tries to improve held-out validation accuracy without using
validation labels for model selection. It tunes regularization, class weighting,
intercept scaling, and decision threshold only from grouped out-of-fold
predictions on the training set.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs")
FEATURE_DIR = BASE / "feature_discovery_stylistic"
ANALYSIS_DIR = FEATURE_DIR / "analysis"
TRAIN_PATH = FEATURE_DIR / "train_data_stylistic_manual_annotated.csv"
VAL_PATH = FEATURE_DIR / "val_data_stylistic_manual_annotated.csv"
KEY_PATH = FEATURE_DIR / "feature_column_key_stylistic_manual.csv"

GRID_PATH = ANALYSIS_DIR / "lasso_grouped_tuning_grid.csv"
COEF_PATH = ANALYSIS_DIR / "lasso_grouped_tuned_coefficients.csv"
HIGHLIGHTED_PATH = ANALYSIS_DIR / "lasso_grouped_tuned_highlighted_features.csv"
PRED_TRAIN_PATH = ANALYSIS_DIR / "train_lasso_grouped_tuned_predictions.csv"
PRED_VAL_PATH = ANALYSIS_DIR / "val_lasso_grouped_tuned_predictions.csv"
METRICS_PATH = ANALYSIS_DIR / "lasso_grouped_tuned_metrics.json"
REPORT_PATH = ANALYSIS_DIR / "lasso_grouped_tuned_report.md"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    key = pd.read_csv(KEY_PATH)
    feature_cols = key["feature_column"].tolist()
    for split_name, df in [("train", train), ("val", val)]:
        missing = sorted(set(feature_cols) - set(df.columns))
        if missing:
            raise ValueError(f"{split_name} is missing feature columns: {missing[:5]}")
    return train, val, key, feature_cols


def make_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = (df["year"].astype(int).to_numpy() == 2025).astype(int)
    return x, y


def make_model(C: float, class_weight: str | None, intercept_scaling: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=C,
            class_weight=class_weight,
            intercept_scaling=intercept_scaling,
            max_iter=10000,
            random_state=29,
        ),
    )


def metrics_from_prob(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_2025": float(precision_score(y_true, pred, zero_division=0)),
        "recall_2025": float(recall_score(y_true, pred, zero_division=0)),
        "f1_2025": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - prob, prob]))),
    }


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 161)
    train_prior = float(y_true.mean())

    def key(threshold: float) -> tuple[float, float, float]:
        pred = (prob >= threshold).astype(int)
        return (
            float(accuracy_score(y_true, pred)),
            float(balanced_accuracy_score(y_true, pred)),
            -abs(float(threshold) - train_prior),
        )

    return float(max(thresholds, key=key))


def evaluate_candidate(
    C: float,
    class_weight: str | None,
    intercept_scaling: float,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | str | None]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=29)
    oof_prob = np.zeros(len(y), dtype=float)
    fold_acc = []
    fold_bal = []
    fold_auc = []
    for train_idx, test_idx in cv.split(x, y, groups):
        model = make_model(C, class_weight, intercept_scaling)
        model.fit(x[train_idx], y[train_idx])
        prob = model.predict_proba(x[test_idx])[:, 1]
        oof_prob[test_idx] = prob
        pred = (prob >= 0.5).astype(int)
        fold_acc.append(float(accuracy_score(y[test_idx], pred)))
        fold_bal.append(float(balanced_accuracy_score(y[test_idx], pred)))
        fold_auc.append(float(roc_auc_score(y[test_idx], prob)))

    threshold = choose_threshold(y, oof_prob)
    oof_pred = (oof_prob >= threshold).astype(int)
    full_model = make_model(C, class_weight, intercept_scaling)
    full_model.fit(x, y)
    coefficients = full_model.named_steps["logisticregression"].coef_[0]
    selected_features = int((np.abs(coefficients) > 1e-8).sum())
    return {
        "C": float(C),
        "class_weight": "none" if class_weight is None else str(class_weight),
        "intercept_scaling": float(intercept_scaling),
        "threshold": threshold,
        "oof_accuracy": float(accuracy_score(y, oof_pred)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, oof_pred)),
        "oof_roc_auc": float(roc_auc_score(y, oof_prob)),
        "fold_accuracy_mean_at_0_5": float(np.mean(fold_acc)),
        "fold_accuracy_se_at_0_5": float(np.std(fold_acc, ddof=1) / np.sqrt(len(fold_acc))),
        "fold_balanced_accuracy_mean_at_0_5": float(np.mean(fold_bal)),
        "fold_roc_auc_mean": float(np.mean(fold_auc)),
        "selected_features": selected_features,
    }


def feature_stats(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    y = (df["year"].astype(int) == 2025).astype(int)
    rows = []
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="raise")
        v20 = values[y == 0]
        v25 = values[y == 1]
        rows.append(
            {
                "feature_column": col,
                f"{prefix}_mean_2020": float(v20.mean()),
                f"{prefix}_mean_2025": float(v25.mean()),
                f"{prefix}_diff_2025_minus_2020": float(v25.mean() - v20.mean()),
                f"{prefix}_support_2020": int((v20 > 0).sum()),
                f"{prefix}_support_2025": int((v25 > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def direction(value: float) -> str:
    if value > 0:
        return "2025"
    if value < 0:
        return "2020"
    return "tie"


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int) -> str:
    shown = df.head(limit)
    if shown.empty:
        return "_None._"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in shown.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.3f}"
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, key, feature_cols = load_inputs()
    x_train, y_train = make_xy(train, feature_cols)
    x_val, y_val = make_xy(val, feature_cols)
    groups = train["abstract_id"].astype(str).to_numpy()

    c_values = np.logspace(-4, 0, 25)
    candidates = [
        (float(C), class_weight, float(intercept_scaling))
        for C in c_values
        for class_weight in [None, "balanced"]
        for intercept_scaling in [1.0, 10.0]
    ]
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_JOBS", "4")))
    n_jobs = max(1, min(n_jobs, len(candidates)))
    grid_rows = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(evaluate_candidate)(C, class_weight, intercept_scaling, x_train, y_train, groups)
        for C, class_weight, intercept_scaling in candidates
    )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["oof_accuracy", "oof_balanced_accuracy", "selected_features", "C"],
        ascending=[False, False, True, True],
    )
    grid.to_csv(GRID_PATH, index=False)
    chosen = grid.iloc[0].to_dict()

    class_weight = None if chosen["class_weight"] == "none" else chosen["class_weight"]
    model = make_model(float(chosen["C"]), class_weight, float(chosen["intercept_scaling"]))
    model.fit(x_train, y_train)
    threshold = float(chosen["threshold"])
    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    train_metrics = metrics_from_prob(y_train, train_prob, threshold)
    val_metrics = metrics_from_prob(y_val, val_prob, threshold)
    train_metrics_default = metrics_from_prob(y_train, train_prob, 0.5)
    val_metrics_default = metrics_from_prob(y_val, val_prob, 0.5)

    train_prior = int(np.bincount(y_train).argmax())
    val_majority_pred = np.full_like(y_val, train_prior)
    majority_baseline = {
        "train_majority_class": "2025" if train_prior == 1 else "2020",
        "validation_accuracy": float(accuracy_score(y_val, val_majority_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, val_majority_pred)),
    }

    coefficients = model.named_steps["logisticregression"].coef_[0]
    coef_df = key.copy()
    coef_df["coef_standardized_log_odds_2025"] = coefficients
    coef_df["abs_coef"] = np.abs(coefficients)
    coef_df["odds_ratio_per_1sd"] = np.exp(np.clip(coefficients, -50, 50))
    coef_df["selected_by_lasso"] = coef_df["abs_coef"] > 1e-8
    coef_df["coefficient_direction"] = coef_df["coef_standardized_log_odds_2025"].map(direction)
    coef_df = coef_df.merge(feature_stats(train, feature_cols, "train"), on="feature_column")
    coef_df = coef_df.merge(feature_stats(val, feature_cols, "val"), on="feature_column")
    coef_df["val_mean_direction"] = coef_df["val_diff_2025_minus_2020"].map(direction)
    coef_df["held_same_direction_on_val"] = (
        coef_df["selected_by_lasso"]
        & (coef_df["coefficient_direction"] != "tie")
        & (coef_df["coefficient_direction"] == coef_df["val_mean_direction"])
    )
    coef_df = coef_df.sort_values(["abs_coef", "feature_id"], ascending=[False, True])
    coef_df.to_csv(COEF_PATH, index=False)
    highlighted = coef_df[coef_df["selected_by_lasso"]].copy()
    highlighted.to_csv(HIGHLIGHTED_PATH, index=False)

    for df, prob, path in [
        (train, train_prob, PRED_TRAIN_PATH),
        (val, val_prob, PRED_VAL_PATH),
    ]:
        pred = (prob >= threshold).astype(int)
        out = df[["sentence", "abstract_id", "year"]].copy()
        out["prob_2025"] = prob
        out["threshold"] = threshold
        out["predicted_year"] = np.where(pred == 1, 2025, 2020)
        out.to_csv(path, index=False)

    selected = int(highlighted.shape[0])
    held = int(highlighted["held_same_direction_on_val"].sum())
    positive = highlighted[highlighted["coef_standardized_log_odds_2025"] > 0]
    negative = highlighted[highlighted["coef_standardized_log_odds_2025"] < 0]
    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "Choose the candidate with highest grouped out-of-fold training accuracy; "
            "break ties by grouped out-of-fold balanced accuracy, then fewer selected features."
        ),
        "validation_labels_used_for_selection": False,
        "grouping": "abstract_id",
        "chosen_candidate": chosen,
        "train_metrics_at_train_cv_threshold": train_metrics,
        "val_metrics_at_train_cv_threshold": val_metrics,
        "train_metrics_at_0_5": train_metrics_default,
        "val_metrics_at_0_5": val_metrics_default,
        "majority_baseline": majority_baseline,
        "selected_features": {
            "total": selected,
            "toward_2025": int((highlighted["coef_standardized_log_odds_2025"] > 0).sum()),
            "toward_2020": int((highlighted["coef_standardized_log_odds_2025"] < 0).sum()),
            "held_same_direction_on_val": held,
            "held_same_direction_rate": float(held / selected) if selected else None,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    cols = [
        "feature_id",
        "feature_name",
        "level",
        "coef_standardized_log_odds_2025",
        "train_diff_2025_minus_2020",
        "val_diff_2025_minus_2020",
        "held_same_direction_on_val",
    ]
    report = [
        "# Grouped-CV Tuned LASSO Year Analysis",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "Target encoding: `1 = 2025`, `0 = 2020`. Positive coefficients predict 2025-style writing; negative coefficients predict 2020-style writing.",
        "",
        "Selection used only the training set. Cross-validation was grouped by `abstract_id`, so sentences from the same abstract do not appear in both train and held-out folds. Validation labels were used only for the final evaluation.",
        "",
        "## Chosen Model",
        "",
        f"- `C`: `{float(chosen['C']):.6g}`",
        f"- `class_weight`: `{chosen['class_weight']}`",
        f"- `intercept_scaling`: `{float(chosen['intercept_scaling']):.3g}`",
        f"- decision threshold from grouped train OOF predictions: `{threshold:.3f}`",
        f"- selected features: `{selected}`",
        "",
        "## Performance",
        "",
        "| model | validation accuracy | validation balanced accuracy | validation ROC AUC | validation F1 2025 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| grouped-CV tuned LASSO | {val_metrics['accuracy']:.3f} | "
            f"{val_metrics['balanced_accuracy']:.3f} | {val_metrics['roc_auc']:.3f} | {val_metrics['f1_2025']:.3f} |"
        ),
        (
            f"| same model, threshold 0.5 | {val_metrics_default['accuracy']:.3f} | "
            f"{val_metrics_default['balanced_accuracy']:.3f} | {val_metrics_default['roc_auc']:.3f} | {val_metrics_default['f1_2025']:.3f} |"
        ),
        (
            f"| train-majority baseline ({majority_baseline['train_majority_class']}) | "
            f"{majority_baseline['validation_accuracy']:.3f} | "
            f"{majority_baseline['validation_balanced_accuracy']:.3f} | n/a | n/a |"
        ),
        "",
        "Compared with the earlier `class_weight=balanced` CV run, this improves validation accuracy by using train-only grouped tuning and an accuracy-oriented threshold. It still does not beat the majority-class baseline, so the honest conclusion is that these stylistic features carry weak held-out year signal under this split.",
        "",
        "## Selected Feature Hold-Up",
        "",
        f"The tuned lasso retained `{selected}` features. `{held}` retained the same coefficient direction in validation mean activations (`{100 * held / selected:.1f}%`).",
        "",
        "## Strongest 2025 Coefficients",
        "",
        markdown_table(positive, cols, 15),
        "",
        "## Strongest 2020 Coefficients",
        "",
        markdown_table(negative, cols, 15),
        "",
        "## Outputs",
        "",
        f"- `{GRID_PATH}`",
        f"- `{COEF_PATH}`",
        f"- `{HIGHLIGHTED_PATH}`",
        f"- `{PRED_TRAIN_PATH}`",
        f"- `{PRED_VAL_PATH}`",
        f"- `{METRICS_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(f"Wrote {REPORT_PATH}")
    print(f"chosen C={float(chosen['C']):.6g}, class_weight={chosen['class_weight']}, threshold={threshold:.3f}")
    print(f"val accuracy={val_metrics['accuracy']:.3f}, val balanced accuracy={val_metrics['balanced_accuracy']:.3f}, val auc={val_metrics['roc_auc']:.3f}")
    print(f"majority baseline val accuracy={majority_baseline['validation_accuracy']:.3f}")


if __name__ == "__main__":
    main()
