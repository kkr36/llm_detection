#!/usr/bin/env python3
"""Use L1-logistic regression to find feature labels predictive of 2020 vs 2025."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
LABELED_CSV = BASE / "feature_discovery_corpus_train_valid_subset_labeled_256.csv"
FEATURE_KEY_CSV = BASE / "feature_column_key.csv"
COEFFICIENTS_CSV = BASE / "lasso_year_feature_coefficients.csv"
SELECTED_CSV = BASE / "lasso_year_selected_features.csv"
HOLDOUT_PREDICTIONS_CSV = BASE / "lasso_year_holdout_predictions.csv"
REPORT_MD = BASE / "lasso_year_prediction_report.md"

ORIGINAL_COLUMNS = [
    "sentence",
    "abstract_id",
    "year",
    "p_llm_m0",
    "p_llm_m1",
    "p_llm_m2",
    "p_llm_m3",
    "p_llm_m4",
    "p_llm_mean_over_models",
    "p_human_mean_over_models",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit L1-penalized logistic regression using manual feature labels to predict year."
    )
    parser.add_argument("--input", type=Path, default=LABELED_CSV)
    parser.add_argument("--feature-key", type=Path, default=FEATURE_KEY_CSV)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--c-count", type=int, default=15)
    parser.add_argument("--stability-splits", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def validate_labeled_frame(df: pd.DataFrame) -> list[str]:
    feature_cols = [f"feature_{i:03d}" for i in range(1, 257)]
    missing_original = [col for col in ORIGINAL_COLUMNS if col not in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_original or missing_features:
        raise ValueError(f"Missing columns: original={missing_original}, features={missing_features[:5]}")
    if list(df.columns[:10]) != ORIGINAL_COLUMNS:
        raise ValueError("The first 10 columns are not the expected original corpus columns.")
    if list(df.columns[10:266]) != feature_cols:
        raise ValueError("Feature columns are not exactly feature_001 through feature_256 in order.")
    years = set(df["year"].unique())
    if years != {2020, 2025}:
        raise ValueError(f"Expected exactly years 2020 and 2025, found {sorted(years)}.")
    values = set()
    for col in feature_cols:
        values.update(df[col].dropna().unique().tolist())
    values = {float(v) for v in values}
    if not values.issubset({0.0, 0.5, 1.0}):
        raise ValueError(f"Unexpected feature label values: {sorted(values)}")
    bad_2025 = int(((df["year"] == 2025) & (df["p_llm_mean_over_models"] >= 0.1)).sum())
    if bad_2025:
        raise ValueError(f"Found {bad_2025} retained 2025 rows with p_llm_mean_over_models >= 0.1.")
    return feature_cols


def make_l1_logistic_cv(cv_folds: int, c_count: int, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(
            Cs=np.logspace(-3, 2, c_count),
            cv=cv_folds,
            penalty="l1",
            solver="liblinear",
            scoring="roc_auc",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
            refit=True,
        ),
    )


def make_l1_logistic(c: float, seed: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        ),
    )


def metric_summary(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy_at_0.5": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_2025_at_0.5": float(f1_score(y_true, y_pred)),
        "tn_2020": int(tn),
        "fp_2025_pred": int(fp),
        "fn_2020_pred": int(fn),
        "tp_2025": int(tp),
    }


def cv_score_summary(model: object) -> dict[str, float]:
    clf = model.named_steps["logisticregressioncv"]
    score_key = clf.classes_[-1]
    scores = clf.scores_[score_key]
    mean_by_c = scores.mean(axis=0)
    best_idx = int(np.argmax(mean_by_c))
    return {
        "best_cv_roc_auc_mean": float(mean_by_c[best_idx]),
        "best_cv_roc_auc_std": float(scores[:, best_idx].std()),
    }


def prevalence_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = df[feature_cols].astype(float)
    active = x.gt(0)
    y2020 = df["year"].eq(2020)
    y2025 = df["year"].eq(2025)
    return pd.DataFrame(
        {
            "feature_column": feature_cols,
            "active_count": active.sum(axis=0).to_numpy(dtype=int),
            "active_count_2020": active.loc[y2020].sum(axis=0).to_numpy(dtype=int),
            "active_count_2025": active.loc[y2025].sum(axis=0).to_numpy(dtype=int),
            "prevalence_2020": active.loc[y2020].mean(axis=0).to_numpy(),
            "prevalence_2025": active.loc[y2025].mean(axis=0).to_numpy(),
            "mean_label_2020": x.loc[y2020].mean(axis=0).to_numpy(),
            "mean_label_2025": x.loc[y2025].mean(axis=0).to_numpy(),
        }
    ).assign(
        prevalence_delta_2025_minus_2020=lambda d: d["prevalence_2025"] - d["prevalence_2020"],
        mean_label_delta_2025_minus_2020=lambda d: d["mean_label_2025"] - d["mean_label_2020"],
    )


def stability_selection(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    feature_cols: list[str],
    splits: int,
    seed: int,
) -> pd.DataFrame:
    if splits <= 0:
        return pd.DataFrame(
            {
                "feature_column": feature_cols,
                "stability_selection_frequency": np.nan,
                "stability_positive_frequency": np.nan,
                "stability_negative_frequency": np.nan,
                "stability_mean_coefficient": np.nan,
            }
        )

    splitter = StratifiedShuffleSplit(n_splits=splits, train_size=0.8, random_state=seed)
    coefs = np.zeros((splits, len(feature_cols)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, (train_idx, _) in enumerate(splitter.split(x, y)):
            model = make_l1_logistic(c, seed + i + 1)
            model.fit(x[train_idx], y[train_idx])
            coefs[i] = model.named_steps["logisticregression"].coef_[0]

    eps = 1e-8
    return pd.DataFrame(
        {
            "feature_column": feature_cols,
            "stability_selection_frequency": (np.abs(coefs) > eps).mean(axis=0),
            "stability_positive_frequency": (coefs > eps).mean(axis=0),
            "stability_negative_frequency": (coefs < -eps).mean(axis=0),
            "stability_mean_coefficient": coefs.mean(axis=0),
        }
    )


def attach_coefficients(
    key: pd.DataFrame,
    prevalence: pd.DataFrame,
    stability: pd.DataFrame,
    feature_cols: list[str],
    coef: np.ndarray,
) -> pd.DataFrame:
    eps = 1e-8
    coef_df = pd.DataFrame(
        {
            "feature_column": feature_cols,
            "standardized_lasso_coefficient": coef,
            "abs_standardized_lasso_coefficient": np.abs(coef),
            "selected_by_lasso": np.abs(coef) > eps,
            "prediction_direction": np.where(coef > eps, "2025", np.where(coef < -eps, "2020", "not_selected")),
            "odds_ratio_per_sd": np.exp(np.clip(coef, -50, 50)),
        }
    )
    out = (
        key.merge(coef_df, on="feature_column", how="right")
        .merge(prevalence, on="feature_column", how="left")
        .merge(stability, on="feature_column", how="left")
    )
    out["rank_by_abs_coefficient"] = out["abs_standardized_lasso_coefficient"].rank(
        method="first", ascending=False
    ).astype(int)
    return out.sort_values(
        ["selected_by_lasso", "abs_standardized_lasso_coefficient", "stability_selection_frequency"],
        ascending=[False, False, False],
    )


def short_description(text: str, width: int = 95) -> str:
    text = " ".join(str(text).split())
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def markdown_feature_table(rows: pd.DataFrame, n: int) -> str:
    cols = [
        "feature_column",
        "prediction_direction",
        "standardized_lasso_coefficient",
        "stability_selection_frequency",
        "prevalence_2020",
        "prevalence_2025",
        "description",
    ]
    table = rows.loc[:, cols].head(n).copy()
    table["description"] = table["description"].map(short_description)
    return table.to_markdown(index=False, floatfmt=".3f")


def write_report(
    df: pd.DataFrame,
    selected: pd.DataFrame,
    all_coefficients: pd.DataFrame,
    holdout_metrics: dict[str, float | int],
    cv_metrics: dict[str, float],
    best_c: float,
    args: argparse.Namespace,
) -> None:
    top_2025 = all_coefficients.query("prediction_direction == '2025'").sort_values(
        "standardized_lasso_coefficient", ascending=False
    )
    top_2020 = all_coefficients.query("prediction_direction == '2020'").sort_values(
        "standardized_lasso_coefficient", ascending=True
    )
    lines = [
        "# Lasso Year Prediction Report\n\n",
        "Target: `year == 2025` versus `year == 2020`.\n\n",
        "Predictors: the 256 manual feature labels only (`feature_001` through `feature_256`). ",
        "The model is L1-penalized logistic regression, i.e. logistic lasso, fit on standardized feature columns with balanced class weights.\n\n",
        "Input and filtering:\n\n",
        f"- Input CSV: `{args.input.name}`\n",
        f"- Rows: {len(df):,}\n",
        f"- Year counts: {df['year'].value_counts().sort_index().to_dict()}\n",
        "- The input already applies the requested filter excluding 2025 rows with `p_llm_mean_over_models >= 0.1`.\n\n",
        "Model selection and evaluation:\n\n",
        f"- Train/test split: stratified, test size {args.test_size:.2f}, seed {args.seed}\n",
        f"- Inner CV folds for lasso C: {args.cv_folds}\n",
        f"- Selected inverse regularization strength `C`: {best_c:.6g}\n",
        f"- CV ROC-AUC at selected `C`: {cv_metrics['best_cv_roc_auc_mean']:.3f} +/- {cv_metrics['best_cv_roc_auc_std']:.3f}\n",
        f"- Holdout ROC-AUC: {holdout_metrics['roc_auc']:.3f}\n",
        f"- Holdout average precision for 2025: {holdout_metrics['average_precision']:.3f}\n",
        f"- Holdout accuracy at 0.5: {holdout_metrics['accuracy_at_0.5']:.3f}\n",
        f"- Holdout balanced accuracy at 0.5: {holdout_metrics['balanced_accuracy_at_0.5']:.3f}\n",
        f"- Holdout F1 for 2025 at 0.5: {holdout_metrics['f1_2025_at_0.5']:.3f}\n",
        f"- Confusion matrix at 0.5: TN={holdout_metrics['tn_2020']}, FP={holdout_metrics['fp_2025_pred']}, FN={holdout_metrics['fn_2020_pred']}, TP={holdout_metrics['tp_2025']}\n\n",
        "Selected features:\n\n",
        f"- Nonzero lasso coefficients in the full-data refit: {len(selected)} of 256\n",
        f"- Stability splits at selected `C`: {args.stability_splits}\n\n",
        "## Strongest 2025 Predictors\n\n",
        markdown_feature_table(top_2025, 20),
        "\n\n## Strongest 2020 Predictors\n\n",
        markdown_feature_table(top_2020, 20),
        "\n\n## Output Files\n\n",
        f"- All feature coefficients: `{COEFFICIENTS_CSV.name}`\n",
        f"- Nonzero selected features: `{SELECTED_CSV.name}`\n",
        f"- Holdout predictions: `{HOLDOUT_PREDICTIONS_CSV.name}`\n",
    ]
    REPORT_MD.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    feature_cols = validate_labeled_frame(df)
    key = pd.read_csv(args.feature_key)

    x_df = df[feature_cols].astype(float)
    x = x_df.to_numpy()
    y = df["year"].eq(2025).astype(int).to_numpy()
    subset_row_idx = np.arange(len(df))

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        subset_row_idx,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    holdout_model = make_l1_logistic_cv(args.cv_folds, args.c_count, args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        holdout_model.fit(x_train, y_train)
    y_prob = holdout_model.predict_proba(x_test)[:, 1]
    holdout_metrics = metric_summary(y_test, y_prob)
    cv_metrics = cv_score_summary(holdout_model)
    best_c = float(holdout_model.named_steps["logisticregressioncv"].C_[0])

    holdout_predictions = df.iloc[idx_test][ORIGINAL_COLUMNS].copy()
    holdout_predictions.insert(0, "subset_row_idx", idx_test)
    holdout_predictions["y_true_2025"] = y_test
    holdout_predictions["p_pred_2025"] = y_prob
    holdout_predictions["predicted_year_at_0.5"] = np.where(y_prob >= 0.5, 2025, 2020)
    holdout_predictions.to_csv(HOLDOUT_PREDICTIONS_CSV, index=False)

    full_model = make_l1_logistic(best_c, args.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        full_model.fit(x, y)
    full_best_c = best_c
    full_coef = full_model.named_steps["logisticregression"].coef_[0]

    prevalence = prevalence_table(df, feature_cols)
    stability = stability_selection(x, y, full_best_c, feature_cols, args.stability_splits, args.seed)
    all_coefficients = attach_coefficients(key, prevalence, stability, feature_cols, full_coef)
    selected = all_coefficients.loc[all_coefficients["selected_by_lasso"]].copy()

    all_coefficients.to_csv(COEFFICIENTS_CSV, index=False)
    selected.to_csv(SELECTED_CSV, index=False)
    write_report(df, selected, all_coefficients, holdout_metrics, cv_metrics, full_best_c, args)

    print(f"rows={len(df)} features={len(feature_cols)}")
    print(f"year_counts={df['year'].value_counts().sort_index().to_dict()}")
    print(f"holdout_roc_auc={holdout_metrics['roc_auc']:.3f}")
    print(f"holdout_balanced_accuracy={holdout_metrics['balanced_accuracy_at_0.5']:.3f}")
    print(f"full_data_selected_C={full_best_c:.6g}")
    print(f"selected_features={len(selected)}")
    print(f"wrote={COEFFICIENTS_CSV}")
    print(f"wrote={SELECTED_CSV}")
    print(f"wrote={HOLDOUT_PREDICTIONS_CSV}")
    print(f"wrote={REPORT_MD}")


if __name__ == "__main__":
    main()
