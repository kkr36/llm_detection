"""
strategy.py
-----------
The coding agent overwrites the strategy function each iteration.
The only three things that change are:
  1. The function definition (rewrite_strategy_v{t}); do not remove old
     rewrite_strategy functions, just add new ones below.
  2. The CURRENT_STRATEGY pointer at the bottom.
  3. the CURRENT_TIMESTEP increment at the bottom (equivalent to t).

FRESH START: only rewrite_strategy_v0 exists and CURRENT_TIMESTEP == 0. Per the
freshness convention (see task_prompt.txt), this directory is a brand-new analysis
regardless of any results_*.csv present. Add rewrite_strategy_v1, v2, ... below as
you iterate on prompts that fool Fast-DetectGPT.

Signature: rewrite_strategy_v{t}(prompt_model, abstract, model_name) -> (rewritten_abstract, model_name)
  prompt_model : callable(system, prompt) -> str   (e.g. openai_oss_query)
  abstract     : the AI-generated abstract to rewrite (str)
  model_name   : label of the LLM that produced `abstract` (e.g. "GPT OSS 120b")
"""
from util import compute_abstract_stats, split_into_sentences, clean_text  # useful for prompts that use a subset of sentences
import numpy as np
import re
import string
from typing import Dict, List
import pdb


def rewrite_strategy_v0(prompt_model, abstract, model_name):
    raise NotImplementedError

# Optional: keep completed prompt functions in a separate (gitignored) strategy_private.py.
# A fresh directory intentionally ships without one.
try:
    from strategy_private import *
except ImportError:
    pass

CURRENT_STRATEGY = rewrite_strategy_v0
CURRENT_TIMESTEP = 0
