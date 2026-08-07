#!/usr/bin/env python3
"""One-SE L1 logistic lasso analysis for 2020 vs 2025 feature annotations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_DIR = Path(__file__).resolve().parent
FEATURE_DIR = ANALYSIS_DIR.parent
TRAIN_PATH = FEATURE_DIR / "train_data_annotated_256_features.csv"
VAL_PATH = FEATURE_DIR / "val_data_annotated_256_features.csv"
FEATURE_METADATA_PATH = FEATURE_DIR / "final_features.json"

FEATURE_COLS = [f"feature_{i:03d}" for i in range(1, 257)]
YEARS = (2020, 2025)
POSITIVE_YEAR = 2025
RANDOM_STATE = 36
TOP_K = 30


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["sentence", "abstract_id", "year", *FEATURE_COLS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    df = df[df["year"].isin(YEARS)].copy()
    df["year"] = df["year"].astype(int)
    df["target_2025"] = (df["year"] == POSITIVE_YEAR).astype(int)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0).astype(float)
    return df


def load_feature_metadata(path: Path) -> dict[str, dict[str, object]]:
    rows = json.loads(path.read_text())
    metadata: dict[str, dict[str, object]] = {}
    for row in rows:
        feature_id = int(row["id"])
        metadata[f"feature_{feature_id:03d}"] = row
    return metadata


def build_cv_model(y: pd.Series) -> Pipeline:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    classifier = LogisticRegressionCV(
        Cs=np.logspace(-2, 2, 25),
        penalty="l1",
        solver="liblinear",
        scoring="roc_auc",
        cv=cv,
        class_weight="balanced",
        max_iter=10000,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("logistic_lasso", classifier)])


def build_final_model(c_value: float) -> Pipeline:
    classifier = LogisticRegression(
        C=c_value,
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("scaler", StandardScaler()), ("logistic_lasso", classifier)])


def one_standard_error_choice(classifier: LogisticRegressionCV) -> dict[str, float]:
    scores = classifier.scores_[1]
    mean_scores = scores.mean(axis=0)
    se_scores = scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    best_index = int(mean_scores.argmax())
    best_auc = float(mean_scores[best_index])
    one_se_threshold = best_auc - float(se_scores[best_index])
    eligible = np.flatnonzero(mean_scores >= one_se_threshold)
    chosen_index = int(eligible[0])
    return {
        "max_auc_c": float(classifier.Cs_[best_index]),
        "max_cv_auc": best_auc,
        "max_cv_auc_se": float(se_scores[best_index]),
        "one_se_threshold_auc": one_se_threshold,
        "chosen_one_se_c": float(classifier.Cs_[chosen_index]),
        "chosen_one_se_cv_auc": float(mean_scores[chosen_index]),
        "chosen_one_se_cv_auc_se": float(se_scores[chosen_index]),
    }


def cv_path_frame(classifier: LogisticRegressionCV, choice: dict[str, float]) -> pd.DataFrame:
    scores = classifier.scores_[1]
    means = scores.mean(axis=0)
    ses = scores.std(axis=0, ddof=1) / math.sqrt(scores.shape[0])
    frame = pd.DataFrame(
        {
            "C": classifier.Cs_,
            "mean_cv_auc": means,
            "se_cv_auc": ses,
            "within_one_se_of_best": means >= choice["one_se_threshold_auc"],
        }
    )
    frame["is_max_auc_c"] = frame["C"].eq(choice["max_auc_c"])
    frame["is_chosen_one_se_c"] = frame["C"].eq(choice["chosen_one_se_c"])
    return frame


def predict_frame(model: Pipeline, df: pd.DataFrame, split: str) -> pd.DataFrame:
    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    labels = (probs >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "split": split,
            "row_position": np.arange(len(df)),
            "abstract_id": df["abstract_id"].to_numpy(),
            "year": df["year"].to_numpy(),
            "target_2025": df["target_2025"].to_numpy(),
            "pred_prob_2025": probs,
            "pred_label_year": np.where(labels == 1, 2025, 2020),
            "sentence": df["sentence"].to_numpy(),
        }
    )


def metrics_for_predictions(preds: pd.DataFrame) -> dict[str, object]:
    y = preds["target_2025"].to_numpy()
    p = preds["pred_prob_2025"].to_numpy()
    labels = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_2020": int((y == 0).sum()),
        "n_2025": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
        "majority_accuracy": float(max((y == 0).mean(), (y == 1).mean())),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "tn_2020_correct": int(tn),
        "fp_2020_as_2025": int(fp),
        "fn_2025_as_2020": int(fn),
        "tp_2025_correct": int(tp),
    }


def safe_auc(y: pd.Series, scores: pd.Series) -> float:
    if scores.nunique(dropna=False) <= 1:
        return math.nan
    return float(roc_auc_score(y, scores))


def direction_label(value: float) -> str:
    if value > 0:
        return "2025"
    if value < 0:
        return "2020"
    return "zero"


def feature_summary(
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    metadata: dict[str, dict[str, object]],
) -> pd.DataFrame:
    coefs = model.named_steps["logistic_lasso"].coef_.ravel()
    train_means = train_df.groupby("year")[FEATURE_COLS].mean().reindex(YEARS)
    val_means = val_df.groupby("year")[FEATURE_COLS].mean().reindex(YEARS)
    train_nonzero = train_df[FEATURE_COLS].gt(0).groupby(train_df["year"]).mean().reindex(YEARS)
    val_nonzero = val_df[FEATURE_COLS].gt(0).groupby(val_df["year"]).mean().reindex(YEARS)

    rows = []
    y_train = train_df["target_2025"]
    y_val = val_df["target_2025"]
    for index, col in enumerate(FEATURE_COLS, start=1):
        coef = float(coefs[index - 1])
        direction = direction_label(coef)
        sign = np.sign(coef)
        train_diff = float(train_means.loc[2025, col] - train_means.loc[2020, col])
        val_diff = float(val_means.loc[2025, col] - val_means.loc[2020, col])
        aligned_train = train_df[col] * sign if sign else train_df[col]
        aligned_val = val_df[col] * sign if sign else val_df[col]
        meta = metadata.get(col, {})
        rows.append(
            {
                "feature_id": index,
                "feature_column": col,
                "selected_by_lasso": bool(abs(coef) > 1e-10),
                "coefficient_scaled": coef,
                "abs_coefficient_scaled": abs(coef),
                "direction": direction,
                "odds_ratio_per_scaled_unit": float(math.exp(coef)) if abs(coef) < 700 else math.nan,
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "level": meta.get("level", ""),
                "confidence": meta.get("confidence", ""),
                "train_mean_2020": float(train_means.loc[2020, col]),
                "train_mean_2025": float(train_means.loc[2025, col]),
                "train_mean_diff_2025_minus_2020": train_diff,
                "val_mean_2020": float(val_means.loc[2020, col]),
                "val_mean_2025": float(val_means.loc[2025, col]),
                "val_mean_diff_2025_minus_2020": val_diff,
                "train_nonzero_rate_2020": float(train_nonzero.loc[2020, col]),
                "train_nonzero_rate_2025": float(train_nonzero.loc[2025, col]),
                "val_nonzero_rate_2020": float(val_nonzero.loc[2020, col]),
                "val_nonzero_rate_2025": float(val_nonzero.loc[2025, col]),
                "train_direction_agrees_with_coef": bool(sign != 0 and np.sign(train_diff) == sign),
                "val_direction_agrees_with_coef": bool(sign != 0 and np.sign(val_diff) == sign),
                "train_single_feature_auc_aligned": safe_auc(y_train, aligned_train),
                "val_single_feature_auc_aligned": safe_auc(y_val, aligned_val),
            }
        )
    coefficients = pd.DataFrame(rows)
    coefficients["val_to_train_abs_diff_ratio"] = np.where(
        coefficients["train_mean_diff_2025_minus_2020"].abs() > 0,
        coefficients["val_mean_diff_2025_minus_2020"].abs()
        / coefficients["train_mean_diff_2025_minus_2020"].abs(),
        np.nan,
    )
    return coefficients.sort_values(["abs_coefficient_scaled", "feature_id"], ascending=[False, True])


def format_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def clean_cell(value: object, max_len: int = 96) -> str:
    text = str(value).replace("\n", " ").replace("|", "/").strip()
    return text[: max_len - 3].rstrip() + "..." if len(text) > max_len else text


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    data = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in data.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                value = format_float(value)
            cells.append(clean_cell(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(metrics: dict[str, dict[str, object]], coefficients: pd.DataFrame, cv_choice: dict[str, float], intercept: float) -> None:
    selected = coefficients[coefficients["selected_by_lasso"]].copy()
    selected_2025 = selected[selected["direction"] == "2025"].sort_values("coefficient_scaled", ascending=False)
    selected_2020 = selected[selected["direction"] == "2020"].sort_values("coefficient_scaled", ascending=True)
    val_agree = int(selected["val_direction_agrees_with_coef"].sum())
    train_agree = int(selected["train_direction_agrees_with_coef"].sum())
    ratio_median = float(selected["val_to_train_abs_diff_ratio"].replace([np.inf, -np.inf], np.nan).median())

    metric_rows = pd.DataFrame([{"split": split, **values} for split, values in metrics.items()])
    metric_columns = ["split", "n", "n_2020", "n_2025", "auc", "average_precision", "accuracy", "balanced_accuracy", "majority_accuracy", "log_loss", "brier"]
    feature_columns = ["feature_id", "direction", "coefficient_scaled", "train_mean_diff_2025_minus_2020", "val_mean_diff_2025_minus_2020", "val_single_feature_auc_aligned", "name", "description"]

    report = f"""# Lasso Year Feature Analysis

Target coding: `2025 = 1`, `2020 = 0`. Coefficients are from an L1-regularized logistic regression fit on standardized feature columns with class-balanced training weights. Positive coefficients point toward 2025 writing; negative coefficients point toward 2020 writing.

## Model

- Train file: `{TRAIN_PATH}`
- Validation file: `{VAL_PATH}`
- Feature metadata: `{FEATURE_METADATA_PATH}`
- Selection rule: one-standard-error lasso choice from stratified train CV, scored by ROC-AUC
- CV max-AUC regularization strength `C`: `{cv_choice["max_auc_c"]:.6g}`
- Max mean train CV ROC-AUC: `{cv_choice["max_cv_auc"]:.4f}` +/- `{cv_choice["max_cv_auc_se"]:.4f}` SE
- One-SE threshold ROC-AUC: `{cv_choice["one_se_threshold_auc"]:.4f}`
- Selected one-SE regularization strength `C`: `{cv_choice["chosen_one_se_c"]:.6g}`
- Mean train CV ROC-AUC at selected `C`: `{cv_choice["chosen_one_se_cv_auc"]:.4f}` +/- `{cv_choice["chosen_one_se_cv_auc_se"]:.4f}` SE
- Intercept: `{intercept:.6g}`
- Nonzero selected features: `{len(selected)}` of `{len(coefficients)}`
- Selected toward 2025: `{len(selected_2025)}`
- Selected toward 2020: `{len(selected_2020)}`

## Split Metrics

{markdown_table(metric_rows, metric_columns)}

Confusion counts use threshold `pred_prob_2025 >= 0.5`. Full counts are in `lasso_2020_vs_2025_metrics.json`.

## Validation Hold-Up

- Train sign agreement among selected features: `{train_agree}/{len(selected)}`
- Validation sign agreement among selected features: `{val_agree}/{len(selected)}`
- Median validation/train absolute mean-difference ratio among selected features: `{ratio_median:.3f}`

Sign agreement means the feature's average activation is higher in the year implied by its train coefficient.

## Strongest 2025 Predictors

{markdown_table(selected_2025, feature_columns, max_rows=TOP_K)}

## Strongest 2020 Predictors

{markdown_table(selected_2020, feature_columns, max_rows=TOP_K)}

## Output Files

- `lasso_2020_vs_2025_cv_path.csv`: train CV ROC-AUC by C and one-SE eligibility
- `lasso_2020_vs_2025_coefficients.csv`: all 256 coefficients and train/validation feature summaries
- `lasso_2020_vs_2025_selected_features.csv`: nonzero lasso-selected features at the one-SE C
- `lasso_2020_vs_2025_train_highlighted_features.csv`: top 30 train predictors in each direction with validation hold-up columns
- `lasso_2020_vs_2025_predictions_train.csv`: train predictions
- `lasso_2020_vs_2025_predictions_val.csv`: validation predictions
- `lasso_2020_vs_2025_metrics.json`: scalar metrics and confusion counts
"""
    (ANALYSIS_DIR / "lasso_2020_vs_2025_report.md").write_text(report)


def main() -> None:
    for stale in ["lasso_2020_vs_2025_stable_features.csv", "lasso_2020_vs_2025_regularization_audit.csv"]:
        stale_path = ANALYSIS_DIR / stale
        if stale_path.exists():
            stale_path.unlink()

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    metadata = load_feature_metadata(FEATURE_METADATA_PATH)

    cv_model = build_cv_model(train_df["target_2025"])
    cv_model.fit(train_df[FEATURE_COLS], train_df["target_2025"])
    cv_classifier = cv_model.named_steps["logistic_lasso"]
    cv_choice = one_standard_error_choice(cv_classifier)
    cv_path_frame(cv_classifier, cv_choice).to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_cv_path.csv", index=False)

    model = build_final_model(cv_choice["chosen_one_se_c"])
    model.fit(train_df[FEATURE_COLS], train_df["target_2025"])
    intercept = float(model.named_steps["logistic_lasso"].intercept_[0])

    train_preds = predict_frame(model, train_df, "train")
    val_preds = predict_frame(model, val_df, "val")
    metrics = {"train": metrics_for_predictions(train_preds), "val": metrics_for_predictions(val_preds)}

    coefficients = feature_summary(model, train_df, val_df, metadata)
    selected = coefficients[coefficients["selected_by_lasso"]].copy()
    highlighted = pd.concat(
        [
            selected[selected["direction"] == "2025"].nlargest(TOP_K, "coefficient_scaled"),
            selected[selected["direction"] == "2020"].nsmallest(TOP_K, "coefficient_scaled"),
        ],
        ignore_index=True,
    )

    train_preds.to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_predictions_train.csv", index=False)
    val_preds.to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_predictions_val.csv", index=False)
    coefficients.to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_coefficients.csv", index=False)
    selected.to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_selected_features.csv", index=False)
    highlighted.to_csv(ANALYSIS_DIR / "lasso_2020_vs_2025_train_highlighted_features.csv", index=False)

    with (ANALYSIS_DIR / "lasso_2020_vs_2025_metrics.json").open("w") as handle:
        json.dump(
            {
                "target": {"2025": 1, "2020": 0},
                "selection_rule": "one-standard-error lasso from stratified train CV",
                "cv_choice": cv_choice,
                "intercept": intercept,
                "n_selected_features": int(len(selected)),
                "n_selected_2025": int((selected["direction"] == "2025").sum()),
                "n_selected_2020": int((selected["direction"] == "2020").sum()),
                "n_train_highlighted_features": int(len(highlighted)),
                "metrics": metrics,
            },
            handle,
            indent=2,
        )

    write_report(metrics, coefficients, cv_choice, intercept)
    print(f"Wrote lasso analysis outputs to {ANALYSIS_DIR}")
    print(f"Selected one-SE C: {cv_choice['chosen_one_se_c']:.6g}")
    print(f"Selected features: {len(selected)} / {len(coefficients)}")
    print(f"Train AUC: {metrics['train']['auc']:.4f}")
    print(f"Val AUC: {metrics['val']['auc']:.4f}")
    print(f"Val balanced accuracy: {metrics['val']['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
