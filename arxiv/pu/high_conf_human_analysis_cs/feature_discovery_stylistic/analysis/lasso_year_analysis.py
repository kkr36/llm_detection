#!/usr/bin/env python3
"""L1-logistic analysis of stylistic features that separate 2020 from 2025."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs")
FEATURE_DIR = BASE / "feature_discovery_stylistic"
ANALYSIS_DIR = FEATURE_DIR / "analysis"
TRAIN_PATH = FEATURE_DIR / "train_data_stylistic_manual_annotated.csv"
VAL_PATH = FEATURE_DIR / "val_data_stylistic_manual_annotated.csv"
KEY_PATH = FEATURE_DIR / "feature_column_key_stylistic_manual.csv"

COEFFICIENTS_PATH = ANALYSIS_DIR / "lasso_coefficients_train.csv"
HIGHLIGHTED_PATH = ANALYSIS_DIR / "highlighted_features_validation.csv"
TOP_2025_PATH = ANALYSIS_DIR / "top_2025_features.csv"
TOP_2020_PATH = ANALYSIS_DIR / "top_2020_features.csv"
PRED_TRAIN_PATH = ANALYSIS_DIR / "train_lasso_predictions.csv"
PRED_VAL_PATH = ANALYSIS_DIR / "val_lasso_predictions.csv"
METRICS_PATH = ANALYSIS_DIR / "lasso_metrics.json"
REPORT_PATH = ANALYSIS_DIR / "lasso_analysis_report.md"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    key = pd.read_csv(KEY_PATH)
    feature_cols = key["feature_column"].tolist()

    expected_years = {2020, 2025}
    for split_name, df in [("train", train), ("val", val)]:
        years = set(df["year"].astype(int).unique())
        if years - expected_years:
            raise ValueError(f"{split_name} contains years outside 2020/2025: {sorted(years)}")
        missing_cols = sorted(set(feature_cols) - set(df.columns))
        if missing_cols:
            raise ValueError(f"{split_name} is missing feature columns: {missing_cols[:5]}")

    return train, val, key, feature_cols


def make_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = (df["year"].astype(int).to_numpy() == 2025).astype(int)
    return x, y


def score_split(y_true: np.ndarray, prob_2025: np.ndarray) -> dict[str, float]:
    pred = (prob_2025 >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob_2025)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision_2025": float(precision_score(y_true, pred, zero_division=0)),
        "recall_2025": float(recall_score(y_true, pred, zero_division=0)),
        "f1_2025": float(f1_score(y_true, pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - prob_2025, prob_2025]))),
        "brier": float(brier_score_loss(y_true, prob_2025)),
    }


def year_feature_stats(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    work = df.copy()
    work["target_2025"] = (work["year"].astype(int) == 2025).astype(int)
    rows = []
    for col in feature_cols:
        values = pd.to_numeric(work[col], errors="raise")
        values_2020 = values[work["target_2025"] == 0]
        values_2025 = values[work["target_2025"] == 1]
        rows.append(
            {
                "feature_column": col,
                f"{prefix}_mean_2020": float(values_2020.mean()),
                f"{prefix}_mean_2025": float(values_2025.mean()),
                f"{prefix}_diff_2025_minus_2020": float(values_2025.mean() - values_2020.mean()),
                f"{prefix}_nonzero_rate_2020": float((values_2020 > 0).mean()),
                f"{prefix}_nonzero_rate_2025": float((values_2025 > 0).mean()),
                f"{prefix}_nonzero_diff_2025_minus_2020": float(
                    (values_2025 > 0).mean() - (values_2020 > 0).mean()
                ),
                f"{prefix}_support_2020": int((values_2020 > 0).sum()),
                f"{prefix}_support_2025": int((values_2025 > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def sign_label(value: float) -> str:
    if value > 0:
        return "2025"
    if value < 0:
        return "2020"
    return "tie"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
    shown = df.head(limit).copy()
    if shown.empty:
        return "_None._"
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in shown.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                value = fmt_float(value)
            cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, key, feature_cols = load_inputs()
    x_train, y_train = make_xy(train, feature_cols)
    x_val, y_val = make_xy(val, feature_cols)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 41),
            cv=cv,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=10000,
            random_state=7,
            refit=True,
        ),
    )
    model.fit(x_train, y_train)

    logistic = model.named_steps["logisticregressioncv"]
    coefficients = logistic.coef_[0]
    selected_c = float(logistic.C_[0])
    intercept = float(logistic.intercept_[0])

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    train_metrics = score_split(y_train, train_prob)
    val_metrics = score_split(y_val, val_prob)

    train_majority = int(np.bincount(y_train).argmax())
    baseline_train_pred = np.full_like(y_train, train_majority)
    baseline_val_pred = np.full_like(y_val, train_majority)
    baseline = {
        "train_majority_class": "2025" if train_majority == 1 else "2020",
        "train_majority_accuracy": float(accuracy_score(y_train, baseline_train_pred)),
        "val_train_majority_accuracy": float(accuracy_score(y_val, baseline_val_pred)),
        "train_majority_balanced_accuracy": float(balanced_accuracy_score(y_train, baseline_train_pred)),
        "val_train_majority_balanced_accuracy": float(balanced_accuracy_score(y_val, baseline_val_pred)),
    }

    coef_df = key.copy()
    coef_df["coef_standardized_log_odds_2025"] = coefficients
    coef_df["abs_coef"] = np.abs(coefficients)
    coef_df["odds_ratio_per_1sd"] = np.exp(np.clip(coefficients, -50, 50))
    coef_df["selected_by_lasso"] = coef_df["abs_coef"] > 1e-8
    coef_df["coefficient_direction"] = coef_df["coef_standardized_log_odds_2025"].map(sign_label)
    coef_df = coef_df.merge(year_feature_stats(train, feature_cols, "train"), on="feature_column")
    coef_df = coef_df.merge(year_feature_stats(val, feature_cols, "val"), on="feature_column")
    coef_df["train_mean_direction"] = coef_df["train_diff_2025_minus_2020"].map(sign_label)
    coef_df["val_mean_direction"] = coef_df["val_diff_2025_minus_2020"].map(sign_label)
    coef_df["held_same_direction_on_val"] = (
        coef_df["selected_by_lasso"]
        & (coef_df["coefficient_direction"] != "tie")
        & (coef_df["coefficient_direction"] == coef_df["val_mean_direction"])
    )
    coef_df["train_val_mean_diff_product"] = (
        coef_df["train_diff_2025_minus_2020"] * coef_df["val_diff_2025_minus_2020"]
    )
    coef_df["validation_abs_diff_retention"] = np.where(
        coef_df["train_diff_2025_minus_2020"].abs() > 0,
        coef_df["val_diff_2025_minus_2020"].abs() / coef_df["train_diff_2025_minus_2020"].abs(),
        np.nan,
    )
    coef_df["lasso_abs_rank"] = coef_df["abs_coef"].rank(method="first", ascending=False).astype(int)

    coef_df = coef_df.sort_values(["abs_coef", "feature_id"], ascending=[False, True])
    highlighted = coef_df[coef_df["selected_by_lasso"]].copy()
    top_2025 = highlighted[highlighted["coef_standardized_log_odds_2025"] > 0].copy()
    top_2020 = highlighted[highlighted["coef_standardized_log_odds_2025"] < 0].copy()

    coef_df.to_csv(COEFFICIENTS_PATH, index=False)
    highlighted.to_csv(HIGHLIGHTED_PATH, index=False)
    top_2025.to_csv(TOP_2025_PATH, index=False)
    top_2020.to_csv(TOP_2020_PATH, index=False)

    train_predictions = train[["sentence", "abstract_id", "year"]].copy()
    train_predictions["prob_2025"] = train_prob
    train_predictions["predicted_year"] = np.where(train_prob >= 0.5, 2025, 2020)
    train_predictions.to_csv(PRED_TRAIN_PATH, index=False)

    val_predictions = val[["sentence", "abstract_id", "year"]].copy()
    val_predictions["prob_2025"] = val_prob
    val_predictions["predicted_year"] = np.where(val_prob >= 0.5, 2025, 2020)
    val_predictions.to_csv(PRED_VAL_PATH, index=False)

    selected_count = int(highlighted.shape[0])
    selected_2025 = int((highlighted["coef_standardized_log_odds_2025"] > 0).sum())
    selected_2020 = int((highlighted["coef_standardized_log_odds_2025"] < 0).sum())
    held_count = int(highlighted["held_same_direction_on_val"].sum())
    held_2025_count = int(top_2025["held_same_direction_on_val"].sum())
    held_2020_count = int(top_2020["held_same_direction_on_val"].sum())

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": {"0": 2020, "1": 2025},
        "model": {
            "type": "L1-penalized logistic regression",
            "selected_c": selected_c,
            "intercept": intercept,
            "cv": "5-fold StratifiedKFold, shuffled, random_state=7",
            "scoring": "roc_auc",
            "class_weight": "balanced",
            "standardization": "StandardScaler fit on train features only",
        },
        "class_balance": {
            "train": {str(k): int(v) for k, v in train["year"].astype(int).value_counts().sort_index().items()},
            "val": {str(k): int(v) for k, v in val["year"].astype(int).value_counts().sort_index().items()},
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "baseline": baseline,
        "selected_features": {
            "total": selected_count,
            "toward_2025": selected_2025,
            "toward_2020": selected_2020,
            "held_same_direction_on_val": held_count,
            "held_same_direction_rate": float(held_count / selected_count) if selected_count else None,
            "toward_2025_held_same_direction": held_2025_count,
            "toward_2020_held_same_direction": held_2020_count,
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
    report = []
    report.append("# Stylistic Feature LASSO Year Analysis\n")
    report.append(f"Generated: `{metrics['generated_at']}`\n")
    report.append("Target encoding: `1 = 2025`, `0 = 2020`. Positive coefficients predict 2025-style writing; negative coefficients predict 2020-style writing.\n")
    report.append("The model is an L1-penalized logistic regression with features standardized on the training set only. Feature highlighting is based only on nonzero training coefficients; validation is used afterward to measure predictive performance and whether those highlighted features preserve the same year direction.\n")

    report.append("## Inputs\n")
    report.append(f"- Train: `{TRAIN_PATH}`")
    report.append(f"- Validation: `{VAL_PATH}`")
    report.append(f"- Feature key: `{KEY_PATH}`\n")

    report.append("## Dataset Balance\n")
    report.append("| split | 2020 rows | 2025 rows | total |")
    report.append("| --- | ---: | ---: | ---: |")
    for split_name, df in [("train", train), ("val", val)]:
        counts = df["year"].astype(int).value_counts()
        report.append(f"| {split_name} | {int(counts.get(2020, 0))} | {int(counts.get(2025, 0))} | {len(df)} |")
    report.append("")

    report.append("## Model Performance\n")
    report.append(f"Chosen inverse regularization strength `C`: `{selected_c:.6g}`.")
    report.append("")
    report.append("| split | ROC AUC | accuracy | balanced accuracy | F1 2025 | log loss |")
    report.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for split_name, split_metrics in [("train", train_metrics), ("val", val_metrics)]:
        report.append(
            "| "
            + split_name
            + " | "
            + " | ".join(
                [
                    fmt_float(split_metrics["roc_auc"]),
                    fmt_float(split_metrics["accuracy"]),
                    fmt_float(split_metrics["balanced_accuracy"]),
                    fmt_float(split_metrics["f1_2025"]),
                    fmt_float(split_metrics["log_loss"]),
                ]
            )
            + " |"
        )
    report.append("")
    report.append(
        f"Train-majority baseline (`{baseline['train_majority_class']}`): "
        f"train accuracy {pct(baseline['train_majority_accuracy'])}, "
        f"validation accuracy {pct(baseline['val_train_majority_accuracy'])}; "
        "balanced accuracy is 50.0% by construction for a single-class predictor.\n"
    )

    report.append("## Training-Selected Features\n")
    report.append(
        f"LASSO retained `{selected_count}` of 256 features: `{selected_2025}` point toward 2025 and `{selected_2020}` point toward 2020. "
        f"On validation, `{held_count}` retained the same mean-activation direction ({pct(held_count / selected_count) if selected_count else 'n/a'})."
    )
    report.append("")
    if selected_2025:
        report.append(
            f"For 2025-positive features, `{held_2025_count}` of `{selected_2025}` held direction on validation "
            f"({pct(held_2025_count / selected_2025)})."
        )
    if selected_2020:
        report.append(
            f"For 2020-positive features, `{held_2020_count}` of `{selected_2020}` held direction on validation "
            f"({pct(held_2020_count / selected_2020)})."
        )
    report.append("")

    report.append("## Strongest 2025 Signals\n")
    report.append(markdown_table(top_2025, cols, limit=15))
    report.append("")

    report.append("## Strongest 2020 Signals\n")
    report.append(markdown_table(top_2020, cols, limit=15))
    report.append("")

    report.append("## Output Files\n")
    for path in [
        COEFFICIENTS_PATH,
        HIGHLIGHTED_PATH,
        TOP_2025_PATH,
        TOP_2020_PATH,
        PRED_TRAIN_PATH,
        PRED_VAL_PATH,
        METRICS_PATH,
    ]:
        report.append(f"- `{path}`")
    report.append("")

    REPORT_PATH.write_text("\n".join(report) + "\n")
    print(f"Wrote {REPORT_PATH}")
    print(f"Validation ROC AUC: {val_metrics['roc_auc']:.3f}")
    print(f"Validation balanced accuracy: {val_metrics['balanced_accuracy']:.3f}")
    print(f"Selected features: {selected_count} ({selected_2025} toward 2025, {selected_2020} toward 2020)")
    print(f"Validation direction hold-up: {held_count}/{selected_count}")


if __name__ == "__main__":
    main()
