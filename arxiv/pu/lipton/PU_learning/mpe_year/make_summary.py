"""
Combine the three methods' per-year MPE estimates into one comparison table.

Reads whatever exists in results/:
    james_mpe.csv    (train 2020 -> eval 2020/2023/2025)
    pu_mpe.csv       (train per year -> eval same year)
    pangram_mpe.csv  (no training -> eval per year)
and writes results/mpe_year_summary.csv (long form) + prints a wide pivot of the
headline AI-fraction estimate per (method, year).

    cd .../PU_learning && python mpe_year/make_summary.py
"""

import os
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# the single headline "estimated AI fraction" column for each method's CSV
HEADLINE = {
    "james_mpe.csv": "mpe",
    "pu_mpe.csv": "mpe",
    "pangram_mpe.csv": "mean_fraction_ai",
}


def main():
    long_rows = []
    for fname, col in HEADLINE.items():
        path = os.path.join(OUT_DIR, fname)
        if not os.path.exists(path):
            print(f"(skip, not found: {fname})")
            continue
        df = pd.read_csv(path)
        if col not in df.columns:
            print(f"(skip, no '{col}' column in {fname}; columns={list(df.columns)})")
            continue
        for _, r in df.iterrows():
            long_rows.append({
                "method": r.get("method", fname.split("_")[0]),
                "year": int(r["year"]),
                "ai_fraction_estimate": r[col],
            })

    if not long_rows:
        print("No results yet. Run run_james.py / eval_pu.py / run_pangram.py first.")
        return

    long_df = pd.DataFrame(long_rows).sort_values(["method", "year"])
    long_path = os.path.join(OUT_DIR, "mpe_year_summary.csv")
    long_df.to_csv(long_path, index=False)

    wide = long_df.pivot_table(index="method", columns="year", values="ai_fraction_estimate")
    print("Estimated AI fraction in held-out human abstracts (by method x year):\n")
    print(wide.to_string())
    print(f"\nSaved -> {long_path}")


if __name__ == "__main__":
    main()
