"""Generate one LaTeX table file per metric from a RAID cross-attack experiment CSV.

Each table has rows keyed by (train_attack, test_attack). For PU (TEDn), the
training and eval attack match (diagonal). For PN, train_attack is always blank.
PN and PU are merged on test_attack so each row shows both methods side-by-side.
"""

import os
import re

import pandas as pd

_DIR = os.path.dirname(__file__)
CSV_PATH_PN = os.path.join(_DIR, "logging_accuracy_raid_attack_pn.csv")
CSV_PATH_TTA = os.path.join(_DIR, "logging_accuracy_raid_attack_tta.csv")
OUTPUT_DIR = os.path.join(_DIR, "latex_tables_attack")

INDEX_COLS = {
    "train_attack", "train_method", "train_alpha", "test_attack", "test_alpha",
    "epochs", "run_id",
}
CI_PAT = re.compile(r".*_[lu]_0\.\d+$")

BOLD = False  # set False to disable bolding of the best value in each row

# Keywords that identify metrics where lower is better (checked as substrings).
_LOWER_KEYWORDS = ("fpr", "fnr", "error", "loss", "brier", "ece", "mce", "calibration", "miss")
# Keywords that identify metrics where closer to 0 is better.
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
    if pd.isna(val):
        return r"\textemdash"
    s = f"{val:.2f}"
    if lo is not None and hi is not None and not pd.isna(lo) and not pd.isna(hi):
        s += f" [{lo:.2f}, {hi:.2f}]"
    if bold:
        s = r"\textbf{" + s + "}"
    return s


def tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def make_table(metric: str, merged: pd.DataFrame, has_ci: bool,
               lo_col: str, hi_col: str, direction: str) -> str:
    m_label = metric.replace("-", "").replace("_", "")
    m_caption = tex_escape(metric)

    col_spec = r"p{3.5cm}p{3.7cm}p{3.7cm}c" if has_ci else r"lccc"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{RAID benchmark results, for different adversarial attacks (training and evaluation \\aigen data balanced between domains, LLMs, decoding strategies, and repetition penalty). We show {m_caption} on LLM outputs with different adversarial attacks applied, with upper and lower bounds attached.}}",
        f"\\label{{tab:cross_{m_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textbf{Eval Attack} & \textbf{Supervised} & \textbf{PU + TTA} & \textbf{PU + TTA Gain} \\",
        r"\midrule",
    ]

    for _, row in merged.iterrows():
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
            f"{tex_escape(row['test_attack'])} & {pn_str} & {pu_str} & {gain_str} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    df = pd.concat(
        [pd.read_csv(CSV_PATH_PN), pd.read_csv(CSV_PATH_TTA)],
        ignore_index=True,
    )

    base_metrics = [
        c for c in df.columns
        if c not in INDEX_COLS and not CI_PAT.match(c)
    ]

    pn = df[df["train_method"] == "PN"]
    pu = df[df["train_method"] == "TEDn"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for metric in base_metrics:
        lo_col = f"{metric}_l_0.95"
        hi_col = f"{metric}_u_0.95"
        has_ci = lo_col in df.columns and hi_col in df.columns

        extra = [lo_col, hi_col] if has_ci else []

        # PN keyed by test_attack only (train_attack is always "none" for PN)
        pn_sel = pn[["test_attack", metric] + extra]
        # PU keyed by (train_attack, test_attack)
        pu_sel = pu[["train_attack", "test_attack", metric] + extra]

        # Outer join on test_attack: each row = one eval attack scenario
        merged = pn_sel.merge(
            pu_sel, on="test_attack", how="outer", suffixes=("_PN", "_PU")
        )

        # Sort: PU rows (with train_attack) first by test_attack, PN-only rows last
        merged["_sort_key"] = merged["train_attack"].fillna("\xff")
        merged = merged.sort_values(
            ["_sort_key", "test_attack"]
        ).drop(columns="_sort_key").reset_index(drop=True)

        direction = metric_direction(metric)
        content = make_table(metric, merged, has_ci, lo_col, hi_col, direction)

        out_path = os.path.join(OUTPUT_DIR, f"table_{metric}.tex")
        with open(out_path, "w") as fh:
            fh.write(content)

    print(f"Wrote {len(base_metrics)} table files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
