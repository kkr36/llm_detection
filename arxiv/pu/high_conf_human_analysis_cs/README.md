# high_conf_human_analysis_cs — dataset (cs.AI, 2020 vs 2025)

Sentence-level LLM-detector predictions on human-submitted arXiv **cs.AI** abstracts
from 2020 and 2025. Dataset only (no analysis yet).

## Source & sampling (`scripts/pipeline.py`, SLURM job 707839, seed 42)
- Abstracts sampled directly from `/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json`
  (the full official abstract), **not** from the rewrite parquets.
- **cs.AI only**: `cs.AI` present as a whitespace-delimited token in `categories`.
- **Year** = `update_date.split("-")[0]` (matches the project's `subsample_by_year.py`).
  `update_date==2020` ⇒ the abstract has not been revised since 2020 ⇒ pre-ChatGPT /
  ground-truth human. 2025 authorship is unknown.
- **Exclusion**: any abstract whose whitespace-normalized 100-char prefix matches a
  `human_abstract` (front-half) in `arxiv_2020_ai_cs._10000_fronthalf.parquet` or
  `arxiv_2025_ai_cs._10000_fronthalf.parquet` is dropped (19,994 prefixes; 3,253 hits).
- Sampled: **2020 = 200** (first 100 → `train`, last 100 → `val`),
  **2025 = 1000** (first 500 → `train`, last 500 → `val`). cs.AI pool sizes available:
  2020 = 5,648, 2025 = 32,796.

## Detectors
Five pretrained PN "all" DistilBERT detectors (seeds 0–4), alpha=0:
`pu/lipton/PU_learning/logging_accuracy_llm/normal_sentence/alpha_0/all_{0..4}/llm_type_all_3/*.pt`.
Convention: **`P(LLM) = softmax(logits)[:, 0]`**, `P(human) = 1 − P(LLM)`.
Sentences via spaCy `en_core_web_lg` (`data_helper.split_into_sentences`).
`abstract_id` = arXiv paper id.

## Deliverable CSVs (`data/`), one set per year × {train, val}
| file | cols | rows(2020 tr/val · 2025 tr/val) |
|---|---|---|
| `data_raw_{year}_{set}.csv` | `sentence`, `abstract_id` | 669/673 · 3941/3855 |
| `all_predictions_{year}_{set}.csv` | `sentence`, `abstract_id`, `p_llm_m0..m4` | same |
| `high_conf_human_{year}_{set}.csv` | `sentence`, `abstract_id` — subset of `data_raw` with mean P(human) over 5 models > 0.9 | 507/541 · 1400/1363 |

## Extra (not required; for downstream use)
- `sampled_abstracts_{year}.csv` — sampled abstracts + arxiv id, categories, update_date, set.
- `sentence_predictions_all.csv` — all sentences, both years/sets, all 5 preds + mean/year/set.

Rerun: `sbatch scripts/run.sbatch` (one 48 GB GPU, `llm_embeddings` env; ~2.5 min).

## Feature-discovery corpora (new 2025 detectors)
Re-scored with the **new** TEDn DistilBERT detectors (5 seeds) in
`/share/garg/arxiv_kaggle/2025_models/ArXiv2025_backhalf_3/*.pt` (P=LLM mirrors,
same `P(LLM)=softmax[:,0]` convention, verified in `scripts/train_2025/`).
Built by `scripts/predict_new_models.py` (SLURM `predict_new_models.sbatch`) then
`scripts/assemble_corpora.py` (CPU).

| file | cols | rows |
|---|---|---|
| `feature_discovery_corpus_train.csv` | sentence, abstract_id, year, p_llm_m0..m4, p_llm_mean_over_models, p_human_mean_over_models | 4610 (2020 tr 669 + 2025 tr 3941) |
| `feature_discovery_corpus_val.csv` | same | 4528 (2020 val 673 + 2025 val 3855) |
| `train_data_unannotated.csv` | sentence, abstract_id, year | 2189 = all 2020 (669) + 2025 with mean P(human)>0.9 (1520) |
| `val_data_unannotated.csv` | sentence, abstract_id, year | 2131 = all 2020 (673) + 2025 with mean P(human)>0.9 (1458) |

The `_unannotated` subsets keep **all 2020 sentences + high-confidence-human 2025
sentences** (mean P(human) over the 5 new models > 0.9) so that downstream feature
discovery compares 2020 human vs high-confidence-2025 human writing. Per-file
intermediate predictions: `preds_{year}_{set}.csv`. See [`instructions.md`](instructions.md).
