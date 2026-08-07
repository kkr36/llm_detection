#!/usr/bin/env python3
"""Unregularized logistic regression analysis for 2020 vs 2025 stylistic features."""

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

GRID_PATH = ANALYSIS_DIR / "logistic_unregularized_grouped_cv.csv"
COEF_PATH = ANALYSIS_DIR / "logistic_unregularized_coefficients.csv"
PRED_TRAIN_PATH = ANALYSIS_DIR / "train_logistic_unregularized_predictions.csv"
PRED_VAL_PATH = ANALYSIS_DIR / "val_logistic_unregularized_predictions.csv"
METRICS_PATH = ANALYSIS_DIR / "logistic_unregularized_metrics.json"
REPORT_PATH = ANALYSIS_DIR / "logistic_unregularized_report.md"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    key = pd.read_csv(KEY_PATH)
    feature_cols = key["feature_column"].tolist()
    return train, val, key, feature_cols


def make_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = (df["year"].astype(int).to_numpy() == 2025).astype(int)
    return x, y


def make_model(class_weight: str | None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=None,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=20000,
            tol=1e-7,
            random_state=31,
        ),
    )


def choose_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 161)
    train_prior = float(y_true.mean())

    def score(threshold: float) -> tuple[float, float, float]:
        pred = (prob >= threshold).astype(int)
        return (
            float(accuracy_score(y_true, pred)),
            float(balanced_accuracy_score(y_true, pred)),
            -abs(float(threshold) - train_prior),
        )

    return float(max(thresholds, key=score))


def split_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_2025": float(precision_score(y_true, pred, zero_division=0)),
        "recall_2025": float(recall_score(y_true, pred, zero_division=0)),
        "f1_2025": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - prob, prob]))),
        "positive_prediction_rate": float(pred.mean()),
    }


def grouped_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    class_weight: str | None,
) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=31)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(x, y, groups):
        model = make_model(class_weight)
        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    return oof


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
            }
        )
    return pd.DataFrame(rows)


def direction(value: float) -> str:
    if value > 0:
        return "2025"
    if value < 0:
        return "2020"
    return "tie"


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int = 15) -> str:
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

    grid_rows = []
    fitted = {}
    for class_weight in [None, "balanced"]:
        label = "none" if class_weight is None else "balanced"
        oof = grouped_oof_predictions(x_train, y_train, groups, class_weight)
        threshold = choose_threshold(y_train, oof)
        model = make_model(class_weight)
        model.fit(x_train, y_train)
        train_prob = model.predict_proba(x_train)[:, 1]
        val_prob = model.predict_proba(x_val)[:, 1]
        for threshold_label, thr in [("0.5", 0.5), ("train_grouped_oof_accuracy", threshold)]:
            row = {
                "class_weight": label,
                "threshold_label": threshold_label,
                "threshold": float(thr),
                "oof_roc_auc": float(roc_auc_score(y_train, oof)),
                "oof_accuracy": float(accuracy_score(y_train, (oof >= thr).astype(int))),
                "oof_balanced_accuracy": float(
                    balanced_accuracy_score(y_train, (oof >= thr).astype(int))
                ),
            }
            row.update({f"val_{k}": v for k, v in split_metrics(y_val, val_prob, thr).items()})
            row.update({f"train_{k}": v for k, v in split_metrics(y_train, train_prob, thr).items()})
            grid_rows.append(row)
        fitted[label] = {
            "model": model,
            "threshold": threshold,
            "train_prob": train_prob,
            "val_prob": val_prob,
            "oof_prob": oof,
        }

    grid = pd.DataFrame(grid_rows).sort_values(
        ["oof_accuracy", "oof_balanced_accuracy"], ascending=[False, False]
    )
    grid.to_csv(GRID_PATH, index=False)
    chosen = grid.iloc[0].to_dict()
    chosen_label = str(chosen["class_weight"])
    chosen_threshold = float(chosen["threshold"])
    chosen_fit = fitted[chosen_label]
    model = chosen_fit["model"]
    train_prob = chosen_fit["train_prob"]
    val_prob = chosen_fit["val_prob"]

    coefficients = model.named_steps["logisticregression"].coef_[0]
    coef_df = key.copy()
    coef_df["coef_standardized_log_odds_2025"] = coefficients
    coef_df["abs_coef"] = np.abs(coefficients)
    coef_df["odds_ratio_per_1sd"] = np.exp(np.clip(coefficients, -50, 50))
    coef_df["coefficient_direction"] = coef_df["coef_standardized_log_odds_2025"].map(direction)
    coef_df = coef_df.merge(feature_stats(train, feature_cols, "train"), on="feature_column")
    coef_df = coef_df.merge(feature_stats(val, feature_cols, "val"), on="feature_column")
    coef_df["val_mean_direction"] = coef_df["val_diff_2025_minus_2020"].map(direction)
    coef_df["held_same_direction_on_val"] = (
        (coef_df["coefficient_direction"] != "tie")
        & (coef_df["coefficient_direction"] == coef_df["val_mean_direction"])
    )
    coef_df = coef_df.sort_values(["abs_coef", "feature_id"], ascending=[False, True])
    coef_df.to_csv(COEF_PATH, index=False)

    for df, prob, path in [
        (train, train_prob, PRED_TRAIN_PATH),
        (val, val_prob, PRED_VAL_PATH),
    ]:
        pred = (prob >= chosen_threshold).astype(int)
        out = df[["sentence", "abstract_id", "year"]].copy()
        out["prob_2025"] = prob
        out["threshold"] = chosen_threshold
        out["predicted_year"] = np.where(pred == 1, 2025, 2020)
        out.to_csv(path, index=False)

    train_majority = int(np.bincount(y_train).argmax())
    val_majority_pred = np.full_like(y_val, train_majority)
    majority_baseline = {
        "train_majority_class": "2025" if train_majority == 1 else "2020",
        "validation_accuracy": float(accuracy_score(y_val, val_majority_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, val_majority_pred)),
    }

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "Unregularized logistic regression, sklearn LogisticRegression(penalty=None, solver='lbfgs')",
        "validation_labels_used_for_selection": False,
        "selection_rule": "Choose class_weight/threshold by highest grouped out-of-fold train accuracy, tie-breaking by grouped balanced accuracy.",
        "grouping": "abstract_id",
        "chosen": chosen,
        "grid": grid.to_dict(orient="records"),
        "majority_baseline": majority_baseline,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    positive = coef_df[coef_df["coef_standardized_log_odds_2025"] > 0]
    negative = coef_df[coef_df["coef_standardized_log_odds_2025"] < 0]
    held = int(coef_df["held_same_direction_on_val"].sum())
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
        "# Unregularized Logistic Regression Year Analysis",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "Target encoding: `1 = 2025`, `0 = 2020`. Positive coefficients predict 2025-style writing; negative coefficients predict 2020-style writing.",
        "",
        "Model selection used only grouped train CV by `abstract_id`; validation labels were only used for final evaluation.",
        "",
        "## Chosen Setup",
        "",
        f"- `class_weight`: `{chosen_label}`",
        f"- threshold: `{chosen_threshold:.3f}` (`{chosen['threshold_label']}`)",
        f"- grouped train OOF accuracy: `{chosen['oof_accuracy']:.3f}`",
        f"- grouped train OOF balanced accuracy: `{chosen['oof_balanced_accuracy']:.3f}`",
        f"- grouped train OOF ROC AUC: `{chosen['oof_roc_auc']:.3f}`",
        "",
        "## Validation Performance",
        "",
        "| setup | validation accuracy | validation balanced accuracy | validation ROC AUC | validation F1 2025 | positive prediction rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in grid.iterrows():
        report.append(
            f"| class_weight={row['class_weight']}, threshold={row['threshold_label']} | "
            f"{row['val_accuracy']:.3f} | {row['val_balanced_accuracy']:.3f} | "
            f"{row['val_roc_auc']:.3f} | {row['val_f1_2025']:.3f} | "
            f"{row['val_positive_prediction_rate']:.3f} |"
        )
    report.extend(
        [
            (
                f"| train-majority baseline ({majority_baseline['train_majority_class']}) | "
                f"{majority_baseline['validation_accuracy']:.3f} | "
                f"{majority_baseline['validation_balanced_accuracy']:.3f} | n/a | n/a | 1.000 |"
            ),
            "",
            "## Coefficient Stability Note",
            "",
            f"All 256 features receive nonzero coefficients in the unregularized fit. `{held}` of 256 coefficient directions match the validation mean-activation direction (`{100 * held / 256:.1f}%`). Because there is no penalty and many stylistic features are correlated, coefficient magnitudes should be treated as descriptive, not as stable feature selection.",
            "",
            "## Largest 2025 Coefficients",
            "",
            markdown_table(positive, cols, 15),
            "",
            "## Largest 2020 Coefficients",
            "",
            markdown_table(negative, cols, 15),
            "",
            "## Outputs",
            "",
            f"- `{GRID_PATH}`",
            f"- `{COEF_PATH}`",
            f"- `{PRED_TRAIN_PATH}`",
            f"- `{PRED_VAL_PATH}`",
            f"- `{METRICS_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(f"Wrote {REPORT_PATH}")
    print(
        f"chosen class_weight={chosen_label}, threshold={chosen_threshold:.3f}, "
        f"val_acc={chosen['val_accuracy']:.3f}, val_bal={chosen['val_balanced_accuracy']:.3f}, "
        f"val_auc={chosen['val_roc_auc']:.3f}"
    )
    print(f"majority baseline val_acc={majority_baseline['validation_accuracy']:.3f}")


if __name__ == "__main__":
    main()
