import re
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from gemini_api import call_gemini_2

# The parquet llm_judge evaluates
TRAIN_DATA_PATH = (
    "/share/garg/arxiv_kaggle/multillm/data_raw/"
    "arxiv_2020_xyz_cs._10000_fronthalf.parquet"
)

# Rewrite columns to evaluate (rewrite_Y omitted)
EVAL_REWRITE_COLS = {
    "rewrite_X", "rewrite_Z",
    "rewrite_Z_1_PN", "rewrite_Z_1_PU",
    "rewrite_Z_2_PN", "rewrite_Z_2_PU",
}


# ── LLM Judge Prompts ─────────────────────────────────────────────────────────

HALLUCINATION_CONTEXT = (
    "You are an expert judge evaluating whether a rewritten text introduces "
    "information that was NOT present in the original text. "
    "Focus only on fabricated or added claims — ignore stylistic differences. "
    "Do not reason at length. Output ONLY a JSON object."
)

HALLUCINATION_PROMPT_TEMPLATE = """\
Original text:
\"\"\"
{original}
\"\"\"

Rewritten text:
\"\"\"
{rewrite}
\"\"\"

Rate how free the rewritten text is from hallucinated or added information not present in the original.
A score of 1.0 means no hallucination at all. A score of 0.0 means the rewrite is entirely composed of fabricated information.
Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}
"""

OMISSION_CONTEXT = (
    "You are an expert judge evaluating whether a rewritten text omits crucial "
    "information that was present in the original text. "
    "Focus only on dropped claims or missing key content — ignore stylistic differences. "
    "Do not reason at length. Output ONLY a JSON object."
)

OMISSION_PROMPT_TEMPLATE = """\
Original text:
\"\"\"
{original}
\"\"\"

Rewritten text:
\"\"\"
{rewrite}
\"\"\"

Rate how completely the rewritten text preserves the crucial information from the original.
A score of 1.0 means nothing important was dropped. A score of 0.0 means the rewrite omits all key information.
Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}
"""


# ── Score extraction ──────────────────────────────────────────────────────────

def extract_score(llm_output: str) -> float | None:
    if not llm_output:
        return None
    json_match = re.search(r'\{[^{}]*"score"\s*:\s*([0-9]*\.?[0-9]+)[^{}]*\}', llm_output)
    if json_match:
        try:
            return max(0.0, min(1.0, float(json_match.group(1))))
        except ValueError:
            pass
    numbers = re.findall(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', llm_output)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
    return None


# ── LLM Judge call ────────────────────────────────────────────────────────────

def judge_pair(original: str, rewrite: str) -> dict:
    """
    Make two LLM judge calls for one (original, rewrite) pair:
      - hallucination_score: penalizes added/fabricated information (0=bad, 1=good)
      - omission_score: penalizes dropped crucial information (0=bad, 1=good)
    """
    hallucination_raw = call_gemini_2(
        HALLUCINATION_CONTEXT,
        HALLUCINATION_PROMPT_TEMPLATE.format(original=original, rewrite=rewrite),
    )
    omission_raw = call_gemini_2(
        OMISSION_CONTEXT,
        OMISSION_PROMPT_TEMPLATE.format(original=original, rewrite=rewrite),
    )
    return {
        "hallucination_raw": hallucination_raw,
        "hallucination_score": extract_score(hallucination_raw),
        "omission_raw": omission_raw,
        "omission_score": extract_score(omission_raw),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Config ────────────────────────────────────────────────────────────────
    parquet_path = TRAIN_DATA_PATH
    output_path = (
        "/share/garg/arxiv_kaggle/multillm/data_raw/"
        "faithfulness_scores_2020_xyz_preds_2.parquet"
    )
    sample_n    = 100
    max_workers = 15
    abstract_col = "human_abstract"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)

    rewrite_cols = [
        c for c in df.columns
        if c.startswith("rewrite") and c in EVAL_REWRITE_COLS
    ]
    print(f"Found {len(rewrite_cols)} rewrite column(s): {rewrite_cols}")

    all_results: list[pd.DataFrame] = []

    for col in rewrite_cols:
        print(f"\n── Judging column: {col} (sample_n={sample_n}) ──")

        valid = df[[abstract_col, col]].dropna()
        if len(valid) > sample_n:
            valid = valid.sample(n=sample_n, random_state=42)

        originals = valid[abstract_col].tolist()
        rewrites  = valid[col].tolist()
        orig_idxs = valid.index.tolist()

        # ── Parallel LLM judge calls ──────────────────────────────────────────
        llm_results = [None] * len(originals)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(judge_pair, orig, rew): i
                for i, (orig, rew) in enumerate(zip(originals, rewrites))
            }
            for fut in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=len(originals),
                desc=col,
            ):
                idx = future_to_idx[fut]
                try:
                    llm_results[idx] = fut.result()
                except Exception as e:
                    print(f"  LLM judge error at index {idx}: {e}")
                    llm_results[idx] = {
                        "hallucination_raw": None,
                        "hallucination_score": None,
                        "omission_raw": None,
                        "omission_score": None,
                    }

        col_df = pd.DataFrame({
            "original":            originals,
            "rewrite":             rewrites,
            "orig_parquet_idx":    orig_idxs,
            "hallucination_raw":   [r["hallucination_raw"]   for r in llm_results],
            "hallucination_score": [r["hallucination_score"] for r in llm_results],
            "omission_raw":        [r["omission_raw"]        for r in llm_results],
            "omission_score":      [r["omission_score"]      for r in llm_results],
        })
        col_df["rewrite_col"] = col
        all_results.append(col_df)

        for score_col, label in [
            ("hallucination_score", "hallucination"),
            ("omission_score",      "omission"),
        ]:
            scored = col_df[score_col].dropna()
            if len(scored):
                print(
                    f"  [{label}] scored {len(scored)}/{len(col_df)} rows | "
                    f"mean={scored.mean():.3f}  median={scored.median():.3f}  "
                    f"std={scored.std():.3f}"
                )

    # ── Combine and save ──────────────────────────────────────────────────────
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows → {output_path}")
    print(f"Run add_pretrained_judge.py --input {output_path} to add model scores.")