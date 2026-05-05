The code for the project ``AI text Detection Is a Moving Target: Test-Time Adaptation Under Continual Distribution Shift'', under submission to NeurIPS 2026.

**Abstract:**

Current approaches for AI text detection often rely on training-time access to labeled datasets of both human-written and AI-generated text. This approach is vulnerable to three types of distribution shifts that occur continually post-deployment and for which labeled data is often unavailable: adversarial humanization, new LLMs being released, and temporal drift in human writing. We propose a test-time adaptation approach (TTA), using positive-unlabeled (PU) learning, that adapts to distribution shifts at inference-time. Empirically, we find that existing state-of-the-art supervised detectors can be systematically, repeatedly evaded by distribution shifts in test-time AI-generated and human writing, while PU learning with test-time adaptation is largely robust {(e.g., the commercial model Pangram detects just 24.1\% of our adversarial AI-generated text, while our PU with TTA approach detects 90.5\%)}. Our results suggest that test-time adaptation is a promising framework for robust AI text detection in the wild.


# llm_detection

Research project for detecting LLM-generated text in academic writing. Covers multiple data sources and a range of detection methodologies.

## Repository Layout

```
llm_detection/
├── arxiv/                  # All experiments on arxiv data (see below)
├── openreview/             # ICLR/OpenReview peer-review analysis
├── ssrn/                   # SSRN abstract embedding and clustering
├── grahams_bounty/         # Graham's bounty dataset experiments
├── latex/                  # Paper/writeup source
├── sample_and_mirror.py    # Sample human abstracts and rewrite with 4 LLMs
├── sample_sentences.py     # Sample human/LLM sentences for inspection
└── mirrored_abstracts.json # Mirrored abstract outputs
```

## arxiv Experiments

> **All experiments on arxiv data live in [`arxiv/`](arxiv/).** Each subdirectory is a self-contained experiment.

| Directory | What it does |
|---|---|
| `arxiv/separation/` | Embeds sentences and runs LDA/PCA to measure geometric separation between human and LLM text |
| `arxiv/pu/` | Positive-Unlabeled learning pipeline (Elkanoto SVC) trained on arxiv abstracts |
| `arxiv/pangram/` | Pangram-word based MLE detection; tracks word frequencies over time to estimate AI adoption rate |
| `arxiv/james_analysis/` | MLE estimator using token-level log-probability ratios (logP/logQ) from human vs. AI distributions |
| `arxiv/james_analysis_20k/` | Scaled-up version of james_analysis at 20k samples |
| `arxiv/james_analysis_code/` | james_analysis variant on code-heavy abstracts |
| `arxiv/james_analysis_llm/` | james_analysis variant using LLM-scored token probabilities |
| `arxiv/xgb_retrain/` | XGBoost yearly pipeline; retrains calibrated detector year-by-year and tracks metrics over time |
| `arxiv/slm_ft/` | Fine-tunes a small language model (RoBERTa) as a binary detector with Platt scaling |
| `arxiv/inference_set_rewrite/` | Adversarial evaluation: iterative prompt rewrites to test detection robustness |
| `arxiv/quick_tests/` | Lightweight sanity checks and exploratory scripts |

## Data

Shared data lives under `/share/garg/arxiv_kaggle/`. Scripts reference it via absolute paths; no local copy needed.

## LLMs Used

Mirror/rewrite experiments use four models: Gemini 3 Pro, Llama 3.3 70b, GPT OSS 120b, Qwen3 80b.
