# High-confidence-human analysis: scientific writing 2020 vs 2025

Do the 5 pretrained LLM-detectors judge 2025 arXiv abstracts as "more AI-written"
than 2020 abstracts — and if so, is **subject area** a confounder (e.g. more
deep-learning content in 2025, which the detectors might reflexively flag as AI)?

## Data
- **2020** (`arxiv_2020_ai_cs._10000_fronthalf_120b_qwen_codex.parquet`): pre-ChatGPT,
  so the submitted `human_abstract` is treated as **ground-truth human**. 100 sampled.
- **2025** (`arxiv_2025_ai_cs._10000_fronthalf.parquet`): authorship **unknown** — the
  submitted `human_abstract` may itself be LLM-assisted. 500 sampled.
- Only the `human_abstract` (real submitted abstract) column is used for both years.
- Abstracts are the **front half** of the official abstract; arXiv subject
  categories are joined from `arxiv-metadata-oai-snapshot.json` on a whitespace-
  normalized 100-char prefix.

## Detectors
Five pretrained DistilBERT PN detectors (seeds 0–4), trained on all-LLM sentence
data at alpha=0:
`pu/lipton/PU_learning/logging_accuracy_llm/normal_sentence/alpha_0/all_{0..4}/llm_type_all_3/*.pt`.

Convention (verified empirically + against `prepare_heatmap.py:164`):
**`P(LLM) = softmax(logits)[:, 0]`**, so `P(human) = 1 - P(LLM)`.

## Method
1. Sample 100 (2020) and 500 (2025) human abstracts (seed 42).
2. Sentence-split each abstract (spaCy `en_core_web_lg`, via `data_helper.split_into_sentences`).
3. Score every sentence with all 5 detectors → per-sentence P(LLM) (all saved).
4. Abstract score = per model, mean P(LLM) over its sentences; then averaged over
   the 5 models. `P(human) = 1 - P(LLM)`.
5. High-confidence human: 2025 abstracts with mean P(human) > 0.9.
6. Descriptive stats, interpretability (word log-odds, sentence length), and the
   subject-area confounder tests.

## How to run
```
sbatch scripts/run.sbatch      # one 48 GB GPU, llm_embeddings env; 3 stages in sequence
```
Stage 1 `pipeline.py` (inference) → Stage 2 `build_categories.py` (category join)
→ Stage 3 `analyze.py` (stats + figures).

## Outputs (`data/`)
| file | contents |
|---|---|
| `2020_human_sample.csv` | 100 sampled 2020 human abstracts + per-model/abstract P(LLM) + category + DL features |
| `2025_abstract_sample.csv` | 500 sampled 2025 abstracts + abstract-level preds + category + DL features |
| `sentence_predictions.csv` | **SINGLE combined file**: every sentence, both years — `year`, `sentence`, `abstract_id`, all 5 model P(LLM) preds (+ mean) |
| `high_conf_human_2025.csv` | 2025 abstracts with mean P(human) > 0.9 |
| `categories.parquet` | prefix_key → arXiv categories |
| `descriptive_stats.csv` | distribution summaries (abstract & sentence level, both years, per model) |
| `threshold_flag_rates.csv` | 2020 false-positive rate & 2025 flag rate vs threshold |
| `confounder_by_coarse_group.csv`, `confounder_by_primary_category.csv` | P(LLM) & sample share per subject, both years |
| `confounder_summary.csv` | overall gap, category-adjusted gap, DL effect on 2020 human text |
| `interp_word_logodds.csv`, `interp_sentence_length_vs_pllm.csv`, `interp_top_{llm,human}_sentences.csv` | interpretability |

## Figures (`figures/`)
- `fig1_pllm_distribution.png` — abstract-level P(LLM) distribution, 2020 human vs 2025
- `fig2_pllm_by_subject.png` — mean P(LLM) by subject area, both years
- `fig3_subject_shift.png` — subject-area mix shift 2020 → 2025
- `fig4_dl_confounder.png` — deep-learning content vs P(LLM) (incl. real 2020 human text)
- `fig5_sentence_distribution.png` — sentence-level P(LLM) distribution, both years

## Findings (SLURM job 705358; N = 100 abstracts for 2020, 500 for 2025; seed 42)

Full write-up in [`traces/FINDINGS.md`](traces/FINDINGS.md). Headlines:

1. **Detectors judge 2025 as far more AI-written than 2020.** Abstract-level mean P(LLM)
   0.098 (2020, ground-truth human) → 0.391 (2025); gap **+0.293**. 2020 has a 0% flag
   rate at 0.5 (well-calibrated on real human text); 2025 flags 34% of abstracts and is
   strongly bimodal at the sentence level (22% of sentences > 0.9) → mixed authorship.
   82/500 2025 abstracts are high-confidence human (mean P(human) > 0.9).
2. **Subject area is NOT the confounder — ~6% of the gap.** Reweighting 2025 to the 2020
   subject mix leaves a 0.275 gap. On genuine 2020 human text, deep-learning content does
   *not* raise P(LLM) (DL 0.093 vs non-DL 0.101; corr −0.07), so the detectors do not
   reflexively flag DL topics. The prompt's hypothesis is refuted; the rise is a genuine
   within-subject writing-register shift.
3. **Interpretability.** LLM-flagged sentences are longer and full of hedged/promotional
   register (*offering, insights, particularly, tailored, enhances*); human-flagged
   sentences carry concrete reference tokens, above all URLs / code links
   (`https`, `com`, `github`).
