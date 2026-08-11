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


# ----------------------------------------------------------------------------
# Shared helpers / style spec used by the v1..v5 rewrite strategies below.
# (Pure functions / constants -- no mutable global state.)
# ----------------------------------------------------------------------------
def _clean(text):
    """Normalize an LLM rewrite: strip whitespace, drop a leading 'Here is ...:'
    meta line and stray wrapping quotes/backticks. Never returns None."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    lines = t.split("\n")
    if lines and len(lines[0]) < 80 and lines[0].rstrip().endswith(":"):
        # e.g. "Rewritten abstract:" / "Here is the revised abstract:"
        t = "\n".join(lines[1:]).strip()
    if len(t) >= 2 and t[0] in "\"'`" and t[-1] in "\"'`":
        t = t[1:-1].strip()
    return t


# A description of the "real arXiv abstract" register we want the model to hit:
# information-dense, bursty sentence rhythm, specific vocabulary, and free of the
# LLM's favorite smooth connective cadence -- all directions that raise genuine
# lexical/structural surprise (i.e. lower Fast-DetectGPT curvature) the way human
# authors naturally do.
HUMAN_VOICE = (
    "Write in the register of a real arXiv abstract authored by domain experts: "
    "dense and information-packed, using concrete technical terminology plus the "
    "specific quantities, symbols, and named methods the source provides. Vary "
    "sentence length sharply -- interleave short declaratives with longer, "
    "syntactically complex sentences. Avoid an evenly-paced connective cadence and "
    "the usual boilerplate openers (no 'Moreover,', 'Furthermore,', 'In this work, "
    "we', 'Overall,', 'Notably,'). Do not hedge or over-explain. "
    "Output ONLY the abstract text as one self-contained block -- no title, no "
    "preamble, no markdown, no surrounding quotes."
)


def rewrite_strategy_v1(prompt_model, abstract, model_name):
    """Direct single-pass humanizing rewrite into the dense expert register."""
    n = len(abstract.split())
    context = "You are a senior researcher revising a colleague's abstract. " + HUMAN_VOICE
    prompt = (
        "Rewrite the following abstract so it reads as if written by a human domain "
        "expert. Preserve every technical claim, method, and result exactly; invent "
        f"nothing. Keep it to roughly {n} words. Do not rely on trivial synonym swaps "
        "-- genuinely re-express the ideas with expert phrasing and non-generic word "
        "choices.\n\nABSTRACT:\n" + abstract
    )
    out = _clean(prompt_model(context, prompt))
    if len(out) < 40:
        out = _clean(abstract)
    return out, model_name


def rewrite_strategy_v2(prompt_model, abstract, model_name):
    """Latent reconstruction: extract a terse skeleton, regenerate from it.
    Breaking the source token chain forces genuinely new phrasing."""
    n = len(abstract.split())
    c1 = "You extract the logical skeleton of scientific abstracts. Output only a terse list."
    p1 = (
        "List the core content of this abstract as 4-7 compressed bullet points "
        "(problem, approach, key mechanism/method, main result, implication). Use "
        "noun-phrase fragments, not full sentences. No commentary.\n\n" + abstract
    )
    outline = _clean(prompt_model(c1, p1))
    if len(outline) < 20:
        outline = abstract
    c2 = "You are a domain expert writing an arXiv abstract from your own notes. " + HUMAN_VOICE
    p2 = (
        f"Using ONLY these notes, compose a coherent abstract of about {n} words in an "
        "expert human voice. Synthesize the wording freshly rather than stitching the "
        "notes together, and preserve every technical fact.\n\nNOTES:\n" + outline
    )
    out = _clean(prompt_model(c2, p2))
    if len(out) < 40:
        out = _clean(abstract)
    return out, model_name


def rewrite_strategy_v3(prompt_model, abstract, model_name):
    """Compress to a telegraphic core, then expand with deliberate burstiness."""
    n = len(abstract.split())
    c1 = "You compress scientific text to its telegraphic core."
    p1 = (
        "Compress this abstract to 2-3 extremely terse, information-dense sentences "
        "capturing the problem, the method, and the main result. Technical, no "
        "filler.\n\n" + abstract
    )
    core = _clean(prompt_model(c1, p1))
    if len(core) < 20:
        core = abstract
    c2 = "You are a domain expert expanding a terse research summary into a full abstract. " + HUMAN_VOICE
    p2 = (
        f"Expand this compressed summary into a complete abstract of about {n} words. "
        "Deliberately vary the rhythm -- some sentences short and punchy, others "
        "intricate -- and add the specificity a human expert would, without inventing "
        "facts.\n\nSUMMARY:\n" + core +
        "\n\n(Source abstract for factual reference only -- do not reuse its wording:\n" + abstract + ")"
    )
    out = _clean(prompt_model(c2, p2))
    if len(out) < 40:
        out = _clean(abstract)
    return out, model_name


def rewrite_strategy_v4(prompt_model, abstract, model_name):
    """Discourse reordering + lexical de-genericization in a single pass."""
    n = len(abstract.split())
    context = (
        "You are a domain expert restructuring an abstract. " + HUMAN_VOICE +
        " Reorder the discourse so it does NOT follow the generic "
        "background-then-method-then-result template: lead with the central technical "
        "idea or the key result, and fold motivation in later. Prefer precise, "
        "lower-frequency technical vocabulary over common paraphrase."
    )
    prompt = (
        "Rewrite this abstract with a restructured, non-linear discourse order and "
        f"heightened lexical specificity, as a human expert would. Keep about {n} words "
        "and preserve all facts and claims.\n\nABSTRACT:\n" + abstract
    )
    out = _clean(prompt_model(context, prompt))
    if len(out) < 40:
        out = _clean(abstract)
    return out, model_name


def rewrite_strategy_v5(prompt_model, abstract, model_name):
    """Draft -> self-critique for machine-like tells -> targeted revision."""
    n = len(abstract.split())
    c1 = "You are a domain expert rewriting an abstract in a natural human voice. " + HUMAN_VOICE
    p1 = (
        f"Rewrite this abstract as a human expert (about {n} words), preserving all "
        "technical content.\n\nABSTRACT:\n" + abstract
    )
    draft = _clean(prompt_model(c1, p1))
    if len(draft) < 40:
        draft = _clean(abstract)
    c2 = "You are a sharp editor who spots machine-like, generic academic prose."
    p2 = (
        "Identify the 3-5 phrasings in this draft that sound most templated, generic, "
        "or evenly-cadenced -- the parts a careful human author would never write. "
        "List them as short notes only, no rewrite.\n\nDRAFT:\n" + draft
    )
    critique = _clean(prompt_model(c2, p2))
    c3 = "You are a domain expert producing the final abstract. " + HUMAN_VOICE
    p3 = (
        "Revise the draft to fix the noted weaknesses so the prose reads unmistakably "
        "like a human expert: sharper word choice, burstier rhythm, no boilerplate "
        f"connectives. Keep about {n} words and preserve all facts.\n\nDRAFT:\n" + draft +
        "\n\nWEAKNESSES:\n" + (critique if len(critique) >= 10 else "(none noted)")
    )
    out = _clean(prompt_model(c3, p3))
    if len(out) < 40:
        out = draft
    return out, model_name


# Optional: keep completed prompt functions in a separate (gitignored) strategy_private.py.
# A fresh directory intentionally ships without one.
try:
    from strategy_private import *
except ImportError:
    pass

# Five candidate rewrite prompts (v1..v5) were added this iteration and each evaluated
# on val via inner_loop.py --timestep 1..5. VAL AUC vs human_ref (baseline 0.953):
#   v1 0.528 (best) | v2 0.819 | v3 0.750 | v4 0.697 | v5 0.539 (mean d 0.32, best d).
# Winner = v1 (direct dense-register rewrite); pointer set to it. CURRENT_TIMESTEP stays
# at 5 (five iterations/strategies defined); a future iteration would add v6 (--timestep 6).
CURRENT_STRATEGY = rewrite_strategy_v1
CURRENT_TIMESTEP = 5
