"""Generate one LaTeX table file per metric from a RAID shift experiment CSV.

Each table has rows keyed by (shift_col, source_val, target_val) and two
value columns: PN and PU (TEDn).  Point estimates are shown with 95% CIs.

For shifts where PU was trained with source="none" (e.g. model shift), each PN
row (train=x, test=y) shows the PU result for (train=none, test=y).  Standalone
PU-only rows (source="none") are suppressed from the table.
"""

import os
import re

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "logging_accuracy_raid_shift_seed_5.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "latex_tables")

INDEX_COLS = {
    "shift_col", "source_val", "target_val", "train_method",
    "train_alpha", "test_alpha", "epochs", "run_id",
}
KEY_COLS = ["shift_col", "source_val", "target_val"]
CI_PAT = re.compile(r".*_[lu]_0\.\d+$")

BOLD = False  # set False to disable bolding of the best value in each row

_LOWER_KEYWORDS = ("fpr", "fnr", "error", "loss", "brier", "ece", "mce", "calibration", "miss")
_ZERO_KEYWORDS: tuple[str, ...] = ()


def metric_direction(metric: str) -> str:
    """Return 'higher', 'lower', or 'zero' depending on what's optimal for the metric."""
    m = metric.lower()
    for kw in _ZERO_KEYWORDS:
        if kw in m:
            return "zero"
    for kw in _LOWER_KEYWORDS:
        if kw in m:
            return "lower"
    return "higher"


def _bold_flags(val_pu, val_pn, direction: str) -> tuple[bool, bool]:
    """Return (bold_pu, bold_pn): True where the value should be bolded."""
    if pd.isna(val_pu) or pd.isna(val_pn):
        return False, False
    if direction == "higher":
        if val_pu > val_pn:
            return True, False
        if val_pn > val_pu:
            return False, True
    elif direction == "lower":
        if val_pu < val_pn:
            return True, False
        if val_pn < val_pu:
            return False, True
    else:  # zero
        if abs(val_pu) < abs(val_pn):
            return True, False
        if abs(val_pn) < abs(val_pu):
            return False, True
    return True, True  # tie


def _pu_gain_str(val_pu, val_pn, direction: str) -> str:
    """Signed gain: positive means PU is better, negative means PN is better."""
    if pd.isna(val_pu) or pd.isna(val_pn):
        return r"\textemdash"
    if direction == "higher":
        gain = val_pu - val_pn
    elif direction == "lower":
        gain = val_pn - val_pu
    else:  # zero
        gain = abs(val_pn) - abs(val_pu)
    return f"{gain:+.2f}"


def fmt(val, lo=None, hi=None, bold=False) -> str:
    """Format metric value with optional 95% CI as 'v [lo, hi]'."""
    if pd.isna(val):
        return r"\textemdash"
    s = f"{val:.2f}"
    if lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi):
        s += f" [{lo:.2f}, {hi:.2f}]"
    if bold:
        s = r"\textbf{" + s + "}"
    return s


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters relevant to this dataset."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def make_table(metric: str, merged: pd.DataFrame, has_ci: bool,
               lo_col: str, hi_col: str, direction: str) -> str:
    m_label = metric.replace("-", "").replace("_", "")
    m_caption = tex_escape(metric)

    col_spec = r"p{1.6cm}p{1.4cm}p{1.4cm}p{2.8cm}p{2.8cm}c" if has_ci else r"p{1.6cm}p{1.4cm}p{1.4cm}ccc"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{RAID benchmark results, for different distribution shifts (training and evaluation \\aigen data filtered to remove adversarial attacks). We show {m_caption} on out-of-distribution LLM outputs, with upper and lower bounds attached.}}",
        f"\\label{{tab:{m_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Shift} & \textbf{Source} & \textbf{Target}"
        r" & \textbf{Supervised} & \textbf{PU + TTA} & \textbf{PU Gain} \\",
        r"\midrule",
    ]

    prev_shift = None
    for _, row in merged.iterrows():
        shift = row["shift_col"]
        if prev_shift is not None and shift != prev_shift:
            lines.append(r"\midrule")
        prev_shift = shift

        val_pn = row.get(f"{metric}_PN", float("nan"))
        val_pu = row.get(f"{metric}_PU", float("nan"))

        bold_pu, bold_pn = _bold_flags(val_pu, val_pn, direction) if BOLD else (False, False)

        if has_ci:
            pn_str = fmt(val_pn,
                         row.get(f"{lo_col}_PN", float("nan")),
                         row.get(f"{hi_col}_PN", float("nan")),
                         bold=bold_pn)
            pu_str = fmt(val_pu,
                         row.get(f"{lo_col}_PU", float("nan")),
                         row.get(f"{hi_col}_PU", float("nan")),
                         bold=bold_pu)
        else:
            pn_str = fmt(val_pn, bold=bold_pn)
            pu_str = fmt(val_pu, bold=bold_pu)

        gain_str = _pu_gain_str(val_pu, val_pn, direction)

        lines.append(
            f"{tex_escape(shift)} & {tex_escape(row['source_val'])}"
            f" & {tex_escape(row['target_val'])}"
            f" & {pn_str} & {pu_str} & {gain_str} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _merge_pu_into_pn(
    pn_sel: pd.DataFrame,
    pu_normal: pd.DataFrame,
    pu_none: pd.DataFrame,
    metric: str,
    extra: list[str],
) -> pd.DataFrame:
    """Left-join PU results onto PN rows.

    PU rows with a real source_val join on KEY_COLS. PU rows where source_val
    was "none" join on [shift_col, target_val] and fill any cells that are
    still NaN after the first join (e.g. model-shift rows).
    """
    join_cols_full = KEY_COLS
    value_cols = [metric] + extra

    # Primary join: PU with matching source_val
    pu_norm_sel = pu_normal[join_cols_full + value_cols] if len(pu_normal) else pd.DataFrame(columns=join_cols_full + value_cols)
    merged = pn_sel.merge(pu_norm_sel, on=join_cols_full, how="left",
                          suffixes=("_PN", "_PU"))

    if not len(pu_none):
        return merged

    # Secondary fill: PU trained on "none", matched by (shift_col, target_val)
    join_cols_none = ["shift_col", "target_val"]
    none_rename = {c: f"{c}_PU" for c in value_cols}
    pu_none_lookup = (
        pu_none[join_cols_none + value_cols]
        .rename(columns=none_rename)
    )

    merged = merged.merge(pu_none_lookup, on=join_cols_none, how="left",
                          suffixes=("", "_fill"))

    for c in value_cols:
        pu_col = f"{c}_PU"
        fill_col = f"{pu_col}_fill"
        if fill_col in merged.columns:
            merged[pu_col] = merged[pu_col].combine_first(merged[fill_col])
            merged = merged.drop(columns=[fill_col])

    return merged


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    base_metrics = [
        c for c in df.columns
        if c not in INDEX_COLS and not CI_PAT.match(c)
    ]

    pn = df[df["train_method"] == "PN"]
    pu_all = df[df["train_method"] == "TEDn"]
    pu_normal = pu_all[pu_all["source_val"] != "none"]
    pu_none = pu_all[pu_all["source_val"] == "none"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for metric in base_metrics:
        lo_col = f"{metric}_l_0.95"
        hi_col = f"{metric}_u_0.95"
        has_ci = lo_col in df.columns and hi_col in df.columns

        extra = [lo_col, hi_col] if has_ci else []

        pn_sel = pn[KEY_COLS + [metric] + extra]

        merged = _merge_pu_into_pn(pn_sel, pu_normal, pu_none, metric, extra)
        merged = merged.sort_values(KEY_COLS).reset_index(drop=True)

        direction = metric_direction(metric)
        content = make_table(metric, merged, has_ci, lo_col, hi_col, direction)

        # if metric == 'tnr': import pdb; pdb.set_trace()

        out_path = os.path.join(OUTPUT_DIR, f"table_{metric}.tex")
        with open(out_path, "w") as fh:
            fh.write(content)

    print(f"Wrote {len(base_metrics)} table files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
