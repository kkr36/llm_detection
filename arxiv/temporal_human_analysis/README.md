# Temporal analysis of human scientific writing: 2020 vs 2025

Do the pretrained LLM-detectors judge 2025 arXiv abstracts as "more AI-written"
than 2020 abstracts — and if so, is **subject area** a confounder (e.g. more
deep-learning content in 2025, which the detectors reflexively flag as AI)?

## Data
- **2020** (`arxiv_2020_ai_cs._10000_fronthalf_120b_qwen_codex.parquet`): pre-ChatGPT,
  so the submitted `human_abstract` is treated as **ground-truth human**. 100 sampled.
- **2025** (`arxiv_2025_ai_cs._10000_fronthalf.parquet`): authorship **unknown**;
  the submitted `human_abstract` may itself be AI-assisted. 500 sampled + labeled.
- Only the `human_abstract` (real submitted abstract) column is used for both years.
- Abstracts are the **front half** of the official abstract; arXiv subject
  categories are joined from `arxiv-metadata-oai-snapshot.json` on a whitespace-
  normalized 100-char prefix.

## Detectors
Five pretrained DistilBERT PN detectors (seeds 0-4), trained on all-LLM sentence
data at alpha=0:
`logging_accuracy_llm/normal_sentence/alpha_0/all_{0..4}/llm_type_all_3/*.pt`.

Convention (verified empirically + against `prepare_heatmap.py:164`):
**`P(LLM) = softmax(logits)[:, 0]`**, so `P(human) = 1 - P(LLM)`.

## Method
1. Sample 100 (2020) and 500 (2025) human abstracts (seed 42).
2. Sentence-split each abstract (spaCy `en_core_web_lg`, via `data_helper.split_into_sentences`).
3. Score every sentence with all 5 detectors -> per-sentence P(LLM).
4. Abstract score = per model, mean P(LLM) over its sentences; then averaged over
   the 5 models. `P(human) = 1 - P(LLM)`.
5. Pseudo-label: from 2025 abstracts with mean P(human) > 0.9, sample 100.
6. Descriptive stats, interpretability (word log-odds, sentence length), and the
   subject-area confounder tests.

## How to run
```
sbatch src/run.sbatch      # GPU node, llm_embeddings env; 3 stages in sequence
```
Stage 1 `pipeline.py` (inference) -> Stage 2 `build_categories.py` (category join)
-> Stage 3 `analyze.py` (stats + figures).

## Outputs (`data/`)
| file | contents |
|---|---|
| `2020_human_sample.csv` | 100 sampled 2020 human abstracts + per-model/abstract P(LLM) + category + DL features |
| `2020_sentence_predictions.csv` | every 2020 sentence + per-model P(LLM) |
| `2025_abstract_predictions.csv` | all 500 2025 abstracts + abstract-level preds + category + DL features |
| `2025_sentence_predictions.csv` | **big file**: every 2025 sentence + per-model P(LLM) |
| `2025_pseudolabel.csv` | 100 abstracts with mean P(human) > 0.9 |
| `categories.parquet` | prefix_key -> arXiv categories |
| `descriptive_stats.csv` | distribution summaries (abstract & sentence level, both years) |
| `threshold_flag_rates.csv` | 2020 false-positive rate & 2025 flag rate vs threshold |
| `confounder_by_coarse_group.csv`, `confounder_by_primary_category.csv` | P(LLM) & sample share per subject, both years |
| `confounder_summary.csv` | overall gap, category-adjusted gap, DL effect on 2020 human text |
| `interp_word_logodds.csv`, `interp_sentence_length_vs_pllm.csv`, `interp_top_{llm,human}_sentences.csv` | interpretability |

## Figures (`figures/`)
- `fig1_pllm_distribution.png` — P(LLM) distribution, 2020 human vs 2025
- `fig2_pllm_by_subject.png` — mean P(LLM) by subject area, both years
- `fig3_subject_shift.png` — subject-area mix shift 2020 -> 2025
- `fig4_dl_confounder.png` — deep-learning content vs P(LLM) (incl. real 2020 human text)

## Findings (N = 100 abstracts for 2020, 800 for 2025; seed 42)

**1. The detectors judge 2025 abstracts as far more AI-written than 2020.**
- 2020 (ground-truth human) mean P(LLM) = **0.098**; 2025 mean = **0.392** (gap **+0.294**).
- At a 0.5 threshold: 2020 false-positive rate = **0%** (0/100), 2025 flag rate = **35%**.
- The detectors are well-calibrated on genuine human text (2020 median P(LLM) 0.077),
  so the 2025 mass at high P(LLM) is a real distributional shift, not miscalibration.
- 2025 is strongly **bimodal** at the sentence level (median 0.226, mean 0.395, 22% of
  sentences > 0.9) — consistent with mixed human/LLM authorship within abstracts.

**2. Subject area is NOT the confounder — it explains only ~7% of the gap.**
- Reweighting 2025 to the 2020 subject-area mix moves 2025 mean P(LLM) only from
  0.392 -> 0.372; the category-adjusted gap is still **0.274** (**6.8%** of the
  increase attributable to subject shift). The rise is a *within-subject* temporal shift.
- The decisive test: on **2020 ground-truth-human** abstracts, deep-learning content
  does **not** raise P(LLM): DL 0.093 vs non-DL 0.101 (corr(P(LLM), DL-keyword count)
  = **-0.07**). Every 2020 subject — including ML/DL (0.115) and Vision (0.103) — sits
  near the human floor. So the detectors do **not** reflexively flag deep-learning
  writing as AI. The hypothesis in the prompt is refuted.
- A DL–P(LLM) association *does* appear in **2025** (DL 0.470 vs non-DL 0.320,
  corr +0.21). Because it is absent among genuine 2020 humans, this is not a detector
  topic-artifact — it more plausibly reflects heavier real LLM use by 2025 authors in
  AI-adjacent fields (Vision 0.507, NLP 0.475, AI 0.463, Robotics 0.436, ML/DL 0.422
  are the most-flagged 2025 areas).

**3. What the detectors key on (interpretability).**
- Sentences flagged as LLM are longer (P(LLM) climbs from 0.15 at <60 chars to ~0.42
  at 180–220 chars) and dense with hedged/promotional register: *offering, insights,
  particularly, highlighting, primarily, specialized, aligns, emerged, trade-offs, cues*.
- Sentences flagged as human are marked by concrete/reference tokens — most strongly
  **URLs and code links** (`https`, `com`, `github`), plus plain verbs (*we consider,
  presented, find, uses, important*). LLM abstracts rarely include repo links.

**Bottom line.** 2025 human-submitted abstracts really do read as much more LLM-like
than 2020's, and this is *not* an artifact of subject-area drift or of the detectors
over-flagging deep-learning topics (they don't, on genuine 2020 human text). The
shift is a genuine, broad change in scientific writing register across essentially all
CS subfields between 2020 and 2025.

_Caveats:_ 2025 "human_abstract" is the submitted abstract, not verified human;
per-subject 2020 cells are small (n as low as 1–2 for AI/Security/Theory), so read
coarse groups, not fine categories; abstracts are front-halves only.
