"""
CPU assembly step (run after predict_new_models.py).

Uses the NEW 2025-model per-sentence predictions (preds_{year}_{set}.csv) to build:

  feature_discovery_corpus_train.csv   (REPLACES the old file; new prediction cols)
  feature_discovery_corpus_val.csv     (NEW)
      cols: sentence, abstract_id, year, p_llm_m0..m4,
            p_llm_mean_over_models, p_human_mean_over_models

  train_data_unannotated.csv
  val_data_unannotated.csv
      subset = { all 2020 sentences } U { 2025 sentences with p(human) > 0.9 }
      (high-confidence-human 2025; goal is to compare 2020 human vs high-conf 2025
       human sentences). cols: sentence, abstract_id, year  (NO prediction columns)

p(human) = 1 - mean over the 5 new models of P(LLM).
"""
import pandas as pd

DATA = "/home/kkr36/llm_detection/arxiv/pu/high_conf_human_analysis_cs/data"
MCOLS = [f"p_llm_m{i}" for i in range(5)]


def load_year_set(year, st):
    df = pd.read_csv(f"{DATA}/preds_{year}_{st}.csv")
    df["year"] = int(year)
    df["p_llm_mean_over_models"] = df[MCOLS].mean(axis=1)
    df["p_human_mean_over_models"] = 1.0 - df["p_llm_mean_over_models"]
    return df


for st in ("train", "val"):
    d20 = load_year_set("2020", st)
    d25 = load_year_set("2025", st)
    corpus = pd.concat([d20, d25], ignore_index=True)[
        ["sentence", "abstract_id", "year"] + MCOLS
        + ["p_llm_mean_over_models", "p_human_mean_over_models"]]
    corpus.to_csv(f"{DATA}/feature_discovery_corpus_{st}.csv", index=False)

    # subset: all 2020 + high-confidence-human 2025 (p_human > 0.9); text-only cols
    keep = corpus[(corpus["year"] == 2020) |
                  (corpus["p_human_mean_over_models"] > 0.9)]
    sub = keep[["sentence", "abstract_id", "year"]].reset_index(drop=True)
    sub.to_csv(f"{DATA}/{st}_data_unannotated.csv", index=False)

    n25_all = int((corpus["year"] == 2025).sum())
    n25_keep = int(((keep["year"] == 2025)).sum())
    print(f"[{st}] corpus={len(corpus)} (2020={len(d20)}, 2025={len(d25)}); "
          f"unannotated={len(sub)} = 2020 {len(d20)} + 2025 high-human {n25_keep}/{n25_all}; "
          f"mean P(LLM) 2020={d20['p_llm_mean_over_models'].mean():.3f} "
          f"2025={d25['p_llm_mean_over_models'].mean():.3f}")

print("DONE")
