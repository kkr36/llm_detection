The code for the project ``AI text Detection Is a Moving Target: Test-Time Adaptation Under Continual Distribution Shift'', under submission to NeurIPS 2026.

**Abstract:**

Current approaches for AI text detection often rely on training-time access to labeled datasets of both human-written and AI-generated text. This approach is vulnerable to three types of distribution shifts that occur continually post-deployment and for which labeled data is often unavailable: adversarial humanization, new LLMs being released, and temporal drift in human writing. We propose a test-time adaptation approach (TTA), using positive-unlabeled (PU) learning, that adapts to distribution shifts at inference-time. Empirically, we find that existing state-of-the-art supervised detectors can be systematically, repeatedly evaded by distribution shifts in test-time AI-generated and human writing, while PU learning with test-time adaptation is largely robust {(e.g., the commercial model Pangram detects just 24.1\% of our adversarial AI-generated text, while our PU with TTA approach detects 90.5\%)}. Our results suggest that test-time adaptation is a promising framework for robust AI text detection in the wild.

## Repository Layout

```
llm_detection/
├── arxiv/                  # All experiments on arxiv data (see below)
├── sample_and_mirror.py    # Sample human abstracts and rewrite with 4 LLMs
└── sample_sentences.py     # Sample human/LLM sentences for inspection
```