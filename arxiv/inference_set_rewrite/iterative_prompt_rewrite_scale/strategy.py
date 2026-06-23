"""
strategy_private.py
-------------------
Private implementations of rewrite_strategy functions. Gitignored.
Imported by strategy.py at runtime to override the public stubs.
"""
from util import compute_abstract_stats, split_into_sentences, clean_text
import numpy as np
import re
import string
from typing import Dict, List
import pdb

def rewrite_strategy_X(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Y(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Z(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Z_1_PU(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Z_1_PN(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Z_2_PU(prompt_model, abstract, model_name):
    raise NotImplementedError

def rewrite_strategy_Z_2_PN(prompt_model, abstract, model_name):
    raise NotImplementedError

try:
    from strategy_private import *
except ImportError:
    pass
