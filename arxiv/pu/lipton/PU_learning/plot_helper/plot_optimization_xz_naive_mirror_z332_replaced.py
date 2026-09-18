"""Paper figures with only the left-panel adversarial bars replaced.

Starts from plot_optimization_xz_naive_mirror.py's paper figure logic, then
substitutes only the left subplot's adversarial humanizing prompt values:
  - PU+TTA     <- TEDn xz332/rewrite_Z_332
  - PNU+TTA    <- PNU  Z_332/rewrite_Z_332
  - Supervised <- PN   X/rewrite_Z_332

The right subplot remains the anti-Pangram 3.2 / iterated-game trajectory.
Outputs to plot_helper/logging_accuracy_xz_naive_mirror_z332_replaced/paper/.
"""

import os

import pandas as pd

import plot_optimization_xz as P
import plot_optimization_xz_naive_mirror as M


OUTPUT_FOLDER = "logging_accuracy_xz_naive_mirror_z332_replaced"
PAPER_OUTPUT_FOLDER = f"{OUTPUT_FOLDER}/paper"
BASE_DIR = ".."
TEDN_Z332_CSV = f"{BASE_DIR}/logging_accuracy_xz_z332.csv"
PNU_Z332_CSV = f"{BASE_DIR}/logging_accuracy_xz_PNU_z332.csv"
PN_Z332_CSV = f"{BASE_DIR}/logging_accuracy_xz_PN_z332.csv"
ID_COLS = {
    "learning_method", "data_type", "train_llm", "eval_llm", "train_alpha",
    "test_alpha", "flip", "clean", "sentence", "epochs", "model_dir",
    "iteration", "eval_parquet", "run_id", "pangram_score_type",
    "pangram_sample_n", "pangram_sample_seed",
}


def _metric_cols(df):
    return [c for c in df.columns if c not in ID_COLS]


def _single_row(df, method, train_llm, eval_llm, train_alpha):
    row = df[
        df["learning_method"].eq(method)
        & df["train_llm"].eq(train_llm)
        & df["eval_llm"].eq(eval_llm)
        & df["train_alpha"].eq(train_alpha)
    ]
    if len(row) != 1:
        raise SystemExit(
            f"Expected exactly one row for {(method, train_llm, eval_llm, train_alpha)}, got {len(row)}"
        )
    return row.iloc[0]


def _replace_metric_values(target_df, target_key, source_row):
    mask = (
        target_df["learning_method"].eq(target_key[0])
        & target_df["train_llm"].eq(target_key[1])
        & target_df["eval_llm"].eq(target_key[2])
        & target_df["train_alpha"].eq(target_key[3])
    )
    if mask.sum() != 1:
        raise SystemExit(f"Expected exactly one target row for {target_key}, got {mask.sum()}")
    for col in _metric_cols(target_df):
        if col in source_row.index:
            target_df.loc[mask, col] = source_row[col]


def _replace_series_index(series, idx, source_row, metric):
    point_col, lower_col, upper_col, _ = P.resolve_cols(metric)
    points, lowers, uppers = (component.copy() for component in series)
    center = 0.5 if metric in ("bbe", "plugin-int") else 0
    points[idx] = source_row[point_col] - center
    lowers[idx] = source_row[lower_col] - center
    uppers[idx] = source_row[upper_col] - center
    return points, lowers, uppers


def _install_left_only_z332_overrides(tedn_row, pn_row):
    original_strategy_values = M._strategy0_plot_values

    def strategy_values_with_left_z332(df, df_pnu, metric):
        values = original_strategy_values(df, df_pnu, metric)
        values["pu_a"] = _replace_series_index(values["pu_a"], 1, tedn_row, metric)
        values["pn_a"] = _replace_series_index(values["pn_a"], 1, pn_row, metric)
        return values

    M._strategy0_plot_values = strategy_values_with_left_z332


def main():
    P.output_folder = OUTPUT_FOLDER
    M.PAPER_OUTPUT_FOLDER = PAPER_OUTPUT_FOLDER
    os.makedirs(PAPER_OUTPUT_FOLDER, exist_ok=True)

    base = pd.read_csv(M.BASE_CSV)
    v2 = pd.read_csv(M.V2_CSV)
    data = M.merge_v2_over_base(base, v2)

    data = P.swap_pangram_tpr_tnr(data)
    data = P.add_accuracy_cols(data)
    data = P.reverse_bias(data)
    data = P.reverse_plugin(data)

    paper_data = M.swap_method_tpr_tnr(data, M.PANGRAM_332_METHOD)
    paper_data = P.add_accuracy_cols(paper_data)

    data_pnu = pd.read_csv(M.PNU_CSV)
    data_pnu = P.add_accuracy_cols(data_pnu)
    data_pnu = P.reverse_bias(data_pnu)
    data_pnu = P.reverse_plugin(data_pnu)

    tedn_z332 = pd.read_csv(TEDN_Z332_CSV)
    tedn_z332 = P.add_accuracy_cols(tedn_z332)
    tedn_z332 = P.reverse_bias(tedn_z332)
    tedn_z332 = P.reverse_plugin(tedn_z332)

    pnu_z332 = pd.read_csv(PNU_Z332_CSV)
    pnu_z332 = P.add_accuracy_cols(pnu_z332)
    pnu_z332 = P.reverse_bias(pnu_z332)
    pnu_z332 = P.reverse_plugin(pnu_z332)

    pn_z332 = pd.read_csv(PN_Z332_CSV)
    pn_z332 = P.add_accuracy_cols(pn_z332)
    pn_z332 = P.reverse_bias(pn_z332)
    pn_z332 = P.reverse_plugin(pn_z332)

    _replace_metric_values(
        data_pnu,
        ("PNU", "Z", "rewrite_Z", 0.25),
        _single_row(pnu_z332, "PNU", "Z_332", "rewrite_Z_332", 0.25),
    )
    _install_left_only_z332_overrides(
        _single_row(tedn_z332, "TEDn", "xz332", "rewrite_Z_332", 0.25),
        _single_row(pn_z332, "PN", "X", "rewrite_Z_332", 0),
    )

    M.make_strategy0_paper_barplot(paper_data, data_pnu, ["tnr"])
    M.make_strategy0_paper_grids(paper_data, data_pnu, M.PAPER_METRICS)


if __name__ == "__main__":
    main()
