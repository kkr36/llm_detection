"""
strategy.py
-----------
The coding agent overwrites the strategy function each iteration.
The only two things that change are:
  1. The function definition (rewrite_strategy_v{t}); do not remove old rewrite_strategy functions, just add new ones below
  2. The CURRENT_STRATEGY pointer at the bottom
  3. the CURRENT_TIMESTEP increment at the bottom (equivalent to t)
"""
from util import compute_abstract_stats, split_into_sentences, clean_text # split sentences is useful for prompts that use only a subset of sentences
import numpy as np
import re
import string
from typing import Dict, List
import pdb

def rewrite_strategy_v0(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v1(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v2(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v3(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v4(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v5(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v6(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v7(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v8(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v9(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_v10(prompt_model, abstract, model_name):
    raise NotImplementedError

try:
    from strategy_private import *
except ImportError:
    pass

CURRENT_STRATEGY = rewrite_strategy_v9
CURRENT_TIMESTEP = 9
