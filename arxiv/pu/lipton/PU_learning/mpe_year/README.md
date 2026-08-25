# mpe_year

Estimate the **proportion of AI-written text in real arXiv abstracts** for years
**2020, 2023, 2025**, with three methods, all evaluated on the same held-out data.

Everything here only *adds* files and *imports* existing building blocks
(`data_helper.IMDb`, `james_methods`, `algorithm`, `helper`, `estimator`,
`baselines`, `model_helper`); no existing file is modified.

## Data & split

Per year: `/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_cs._10000_fronthalf.parquet`
(10,000 rows). Each row has `human_abstract` plus exactly one of 4 LLM *mirror*
rewrites (round-robin over `Llama 3.3 70b Instruct`, `Gemini 3 Preview`,
`GPT OSS 120b`, `Gemini 2.5 Flash`).

- **Train** = first **8,000** rows (mirrors = AI positives, `human_abstract` = negatives/unlabeled).
- **Eval**  = last **2,000** rows, scored on the **`human_abstract`** column only.

**Unit of analysis = fixed 75-word chunks.** To make the three methods directly
comparable, every method is trained/evaluated on the *identical* pieces of text.
Each abstract (human and AI) is whitespace-split into consecutive 75-word chunks,
with the trailing remainder **merged into the last chunk** (last chunk = 75–149
words; an abstract < 75 words is one chunk), so no text is discarded. The held-out
test set is exactly `chunk_75(human_abstract)` over the last 2,000 rows, and
Pangram / PU / James all score that same chunk set (`eval_human_chunks(year)`).

Split + chunking logic lives in [`mpe_data.py`](mpe_data.py) (`chunk_75`,
`chunks_of`, `chunk_tokens_of`, `eval_human_chunks`, `read_fronthalf_pu`). This is
a deterministic first-8000 / last-2000 split of the plain `_fronthalf` parquet.

> **Eval-set note.** The existing James/PU/xy prepare scripts also hold out the
> *tail* rows and score `human_abstract`, but they `sample(frac=1, random_state=seed)`
> the frame **first** (so it's the last rows of a per-seed shuffle) and use the
> `_120b_qwen`/`xyz` variant parquets, which don't exist for 2023/2025. This folder
> uses the same *concept* (held-out tail, `human_abstract`) with a simpler
> deterministic split so all three years are directly comparable.

## Methods

| Method  | Unit          | Train                              | Eval                              |
|---------|---------------|------------------------------------|-----------------------------------|
| Pangram | 75-word chunk | none (API)                         | shared human chunks / year        |
| James   | 75-word chunk | **once on 2020**                   | 2020/2023/2025 shared human chunks|
| PU/TEDn | 75-word chunk | **per year** (2020, 2023, 2025)    | same year's shared human chunks   |

All three score the **identical** `eval_human_chunks(year)` set.

All three estimate the AI fraction (mixture proportion). Higher over time = more
AI-written abstracts.

## How to run

Launch everything from the `PU_learning` root.

### James (CPU, slurm)
Trains the word log-prob distribution once on 2020, then estimates the AI fraction
on each year's held-out human chunks.
```
sbatch mpe_year/run_james.sbatch          # -> results/james_mpe.csv
```
Env: `llm_embeddings` (spaCy `en_core_web_lg` + `swifter`).

### PU / TEDn (GPU, slurm array)
One model per (year, seed); 3 years x 5 seeds. Models are saved under
`/share/garg/arxiv_kaggle/mpe_year_models/{year}/`.
```
sbatch mpe_year/run_pu_train.sbatch       # trains + saves .pt models
python  mpe_year/eval_pu.py               # -> results/pu_mpe.csv  (needs a GPU)
```
`train_pu.py` uses `alpha=0` (unlabeled pool = pure `human_abstract`), 75-word
chunks, `--clean`, DistilBert, AdamW lr=1e-5, 3 epochs — matching
`scripts/train_2025/train_2025_tedn.py`. `eval_pu.py` uses `BBE_estimator`
(P = val-split mirror chunks, U = eval human chunks) and averages over seeds.

### Pangram (API, **not run automatically** — ~1 call per chunk)
```
/home/kkr36/.conda/envs/llm_master/bin/python mpe_year/run_pangram.py
# -> results/pangram_raw/pangram_raw_{year}.csv  (full raw API response per abstract)
# -> results/pangram_mpe.csv                     (per-year summary)
```
`parse_result` stores the entire raw JSON plus every top-level scalar field, so new
API fields are captured automatically.

### Combine & plot
```
python mpe_year/make_summary.py           # -> results/mpe_year_summary.csv (+ pivot)
python mpe_year/plot_mpe_year.py          # -> results/mpe_year_plot.{pdf,png}
```
`plot_mpe_year.py` draws predicted % AI vs. year, one connected line per method
(skips any method whose CSV isn't there yet). Options: `--pu-metric {mpe,mean_ai_prob}`,
`--pangram-metric {mean_fraction_ai,frac_pred_ai}`, `--overlay-alt`.

## Files
- `mpe_data.py`     — data loading, train/eval split, mirror extraction, 75-word chunking, PU reader
- `run_james.py` / `run_james.sbatch`     — James's method (train 2020, eval all years)
- `train_pu.py` / `run_pu_train.sbatch`   — PU/TEDn training, one model per (year, seed)
- `eval_pu.py`     — PU MPE on held-out human chunks
- `run_pangram.py` — Pangram over held-out human chunks (run manually)
- `make_summary.py`— merge the three methods into one table
- `results/`       — all outputs
