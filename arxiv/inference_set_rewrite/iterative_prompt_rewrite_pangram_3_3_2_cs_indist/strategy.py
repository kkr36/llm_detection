"""Rewrite strategies for the in-distribution (multillm-'cs.') Pangram-3.3.2 optimization.

Fresh t=0 workspace. rewrite_function_1 is the reference prompt (the one that beats
Pangram 3.3.2 ~30% of the time on in-distribution CS abstracts) — byte-faithful copy of
rewrite_strategy_v9_reference / rewrite_strategy_Z.
"""

import hashlib
import json
import os
import random
import re

from util import split_into_sentences

# --- Held-out human-abstract pool for few-shot style emulation (t>=7) ---------
# Built by build_heldout_human_pool.py. The pool is globally disjoint from all 75
# train `original` abstracts (excluded by normalized full-text and 150-char
# prefix), so for any row the few-shot examples cannot contain that row's own
# original. This directly overrides the earlier no-human-few-shot protocol, so
# t>=7 mirrors are an exploratory arm, not "valid" successes under that protocol.
_HELDOUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "heldout_human_abstracts.json"
)
try:
    with open(_HELDOUT_PATH) as _fh:
        _HUMAN_POOL = [it["abstract"] for it in json.load(_fh)]
except FileNotFoundError:
    _HUMAN_POOL = []


def _human_examples(abstract, k):
    """Deterministically pick k held-out human abstracts as style references.

    Seeded by the input text hash: reproducible, thread-safe (local RNG), and not
    routed on arxiv id/title/entity. Corpus-general -- every abstract gets a
    pseudo-random draw from the same disjoint human pool.
    """
    if not _HUMAN_POOL:
        raise RuntimeError(
            "held-out human pool missing; run build_heldout_human_pool.py first"
        )
    seed = int(hashlib.sha1(abstract.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    return rng.sample(_HUMAN_POOL, min(k, len(_HUMAN_POOL)))


def _format_examples(examples):
    return "\n\n".join(
        f"[Human-written abstract {i + 1}]\n{ex}" for i, ex in enumerate(examples)
    )


# Held-out human SENTENCE bank, built once from the pool by a light regex split.
# Used by t=12..t=14 to anchor cadence at sentence granularity -- the only level at
# which human signal helped (v9's 2 rescues came from human anchors inside v1's
# per-sentence atomization, not whole-abstract "rewrite in this style").
_HUMAN_SENTENCES = []
for _ab in _HUMAN_POOL:
    for _s in re.split(r"(?<=[.!?])\s+", _ab.strip()):
        _s = _s.strip()
        if 4 <= len(_s.split()) <= 40:
            _HUMAN_SENTENCES.append(_s)


def _human_sentences(abstract, k):
    """Deterministically draw k held-out human sentences as cadence templates.

    Seeded by the input hash (distinct salt from _human_examples), thread-safe,
    corpus-general. Sentences come from the disjoint human pool, so none is drawn
    from the current row's own original.
    """
    if not _HUMAN_SENTENCES:
        raise RuntimeError(
            "held-out human sentence bank empty; run build_heldout_human_pool.py first"
        )
    seed = int(hashlib.sha1(("SENT::" + abstract).encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed)
    return rng.sample(_HUMAN_SENTENCES, min(k, len(_HUMAN_SENTENCES)))


# Reusable v1 step instructions (plain strings, no interpolation), so the t>=12
# variants can share v1's known-good rejoin / factual-audit / cleanup passes
# without re-deriving them. v1 itself keeps its own inline copies for provenance.
_BANS_LINE = (
    'Avoid these formulaic phrases: "we propose", "we present", "we introduce", '
    '"we demonstrate", "novel", "state-of-the-art", "extensive", "robust", '
    '"in this paper", "in this work", "furthermore", "moreover", "it is worth noting", '
    '"leverages", "this approach", "the proposed", "notably", "significantly", '
    '"outperforms", "facilitate", "enable", "surpasses", "validate", "verify", '
    '"highlight", "thus", "thereby", "utilize", "employ".'
)

_CTX_REJOIN = """
The following is a numbered list of sentences written independently (some short, some long).
Join them into a single coherent abstract paragraph. Make ONLY the minimal edits needed for coherence:
- Fix pronoun or reference ambiguities introduced by joining.
- Fix obvious grammar errors at sentence boundaries.
- Do NOT add transitions, smooth style, or homogenize sentence lengths.
- Preserve every short sentence as-is -- do not expand or merge them.
- Do not use markdown. Do NOT use LaTeX; keep all math in plain Unicode characters.
Output only the joined abstract text.
"""

_CTX_FACTUAL = """
Compare the DRAFT abstract to the SOURCE abstract. Your ONLY task is to correct factual errors in DRAFT:
- If a numeric value in DRAFT is WRONG (not just missing), replace it with the correct value.
- If a named method, dataset, or model in DRAFT is WRONG (misidentified or misspelled), correct it.
- If DRAFT merely OMITS content that is in SOURCE, do NOT add it. Do NOT add new sentences.
- Do NOT add structural descriptions, section summaries, or organizational content.
- Do NOT smooth, polish, or improve the writing. Write "use", not "utilize"/"employ".
- Do NOT use markdown or LaTeX.
If there are no factual errors, return DRAFT unchanged. Output only the (possibly corrected) abstract.
"""

_CTX_JOIN_FACTUAL = """
You are given a numbered list of independently written sentences and the SOURCE abstract they derive from. Do two things:
A. Join the sentences into one coherent paragraph. Make only minimal coherence edits (pronoun/reference ambiguity, boundary grammar). Do NOT add transitions, smooth style, or homogenize lengths; preserve every short sentence.
B. Correct ONLY factual errors against SOURCE: a wrong numeric value, or a misidentified method/dataset/model. Do NOT add omitted content, new sentences, or section summaries.
Do not use markdown or LaTeX; keep math in Unicode. Output only the final abstract paragraph.
"""

_CTX_CLEANUP = """
Proofread the following scientific abstract:
1. Remove any meta-commentary, preamble, or title (e.g. "Here is the abstract:").
2. Fix only clear grammatical errors that impede reading.
3. Do NOT smooth, polish, add transitions, or homogenize the style.
4. Preserve short sentences and all stylistic irregularities.
5. Remove any markdown formatting if present.
Output only the clean abstract text.
"""


def _strip_latex(text):
    """Remove inline/display math delimiters before sending to the model."""
    text = re.sub(r'\\\(', '', text)
    text = re.sub(r'\\\)', '', text)
    text = re.sub(r'\\\[', '', text)
    text = re.sub(r'\\\]', '', text)
    return text


def _strip_latex_post(final):
    """Strip any LaTeX that slipped back into the model output."""
    final = re.sub(r'\\[\(\[](.*?)\\[\)\]]', lambda m: m.group(1), final, flags=re.DOTALL)
    final = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', final)
    return final


def rewrite_function_1(prompt_model, abstract, model_name):
    """Reference prompt: per-sentence alternating rephrase + minimal rejoin + no-restore
    factual audit + cleanup. Applied to a naive AI mirror (mirror_0) to humanize it."""
    # Python preprocessing: strip LaTeX math delimiters
    abstract_clean = re.sub(r'\\\(', '', abstract)
    abstract_clean = re.sub(r'\\\)', '', abstract_clean)
    abstract_clean = re.sub(r'\\\[', '', abstract_clean)
    abstract_clean = re.sub(r'\\\]', '', abstract_clean)

    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))

    # Step 1: Per-sentence alternating rephrase
    context1 = f"""
    Below is a numbered list of sentences from a scientific abstract.
    Rephrase each sentence independently — treat each as a standalone statement.
    Do not use any markdown formatting (no bold, no italics, no bullet points, no headers).
    Do NOT use LaTeX notation or math delimiters (no \\(...\\), no \\[...\\], no \\mathrm{{...}}, no \\subseteq, no \\times).
    Represent mathematical symbols in plain Unicode (e.g., write f⊆X×Y×Z, not \\(f\\subseteq X\\times Y\\times Z\\)).

    ALTERNATING LENGTH RULE (strictly enforced):
    - Odd-numbered sentences (1, 3, 5, ...): write a SHORT version — under 10 words.
      State only the single core fact or claim. Drop all hedging, context, and elaboration.
    - Even-numbered sentences (2, 4, 6, ...): write a FULL technical sentence.
      Vary the syntactic structure from the original (e.g., change active↔passive, move subject).

    For ALL sentences:
    - At most 1 three-word phrase (3-gram) may overlap with the original sentence. Do NOT copy clauses.
    - Preserve all numerical values, named methods, datasets, and model names exactly.
    - Use first-person ("We", "our") naturally where the original does — for example,
      "We show", "We find", "We use", "We examine", "our results", "our method",
      "our analysis". Scientific action verbs are encouraged; only the formulaic phrases below are banned.
    - Avoid ONLY these specific formulaic phrases: "we propose", "we present", "we introduce",
      "we demonstrate", "novel", "state-of-the-art", "extensive", "robust",
      "in this paper", "in this work", "furthermore", "moreover", "it is worth noting",
      "leverages", "this approach", "the proposed", "notably", "significantly",
      "outperforms", "facilitate", "enable", "surpasses", "validate", "verify",
      "highlight", "thus", "thereby", "utilize", "employ",
      "we outline", "we survey", "we provide a", "we provide an", "we offer a", "we offer an".

    Output only the numbered list of rephrases, one per line.
    """
    prompt1 = f"Sentences:\n{numbered}"
    rephrased = prompt_model(context1, prompt1)

    # Step 2: Minimal rejoin
    context2 = """
    The following is a numbered list of sentences written independently (some short, some long).
    Join them into a single coherent abstract paragraph.
    Make ONLY the minimal edits needed for coherence:
    - Fix pronoun or reference ambiguities introduced by joining.
    - Fix obvious grammar errors at sentence boundaries.
    - Do NOT add transitions, smooth style, or homogenize sentence lengths.
    - Preserve every short sentence as-is — do not expand or merge them.
    - Do not use any markdown formatting (no bold, no italics, no bullet points).
    - Do NOT use LaTeX notation or math delimiters. Keep all math in plain Unicode characters.
    Output only the joined abstract text.
    """
    prompt2 = f"Sentences:\n{rephrased}"
    joined = prompt_model(context2, prompt2)

    # Step 3: Factual error correction only — no content insertion (NO omission restore)
    context3 = f"""
    Compare the DRAFT abstract to the ORIGINAL abstract.
    Your ONLY task is to correct factual errors in DRAFT:
    - If a specific numerical value in DRAFT is WRONG (not just missing), replace it with the correct value.
    - If a named method, dataset, or model in DRAFT is WRONG (misidentified or misspelled), correct it.
    - If DRAFT is merely INCOMPLETE (missing content that is in ORIGINAL), do NOT add it.
    - Do NOT add new sentences under any circumstances.
    - Do NOT add structural descriptions, section summaries, or organizational content.
    - Do NOT smooth, polish, or improve the writing.
    - Do NOT use LaTeX notation or math delimiters.
    - Do not write "utilize" or "employ" — write "use" instead.
    - Do not use any markdown formatting.
    If there are no factual errors, return the DRAFT unchanged.
    Output only the (possibly corrected) abstract.
    """
    prompt3 = f"DRAFT:\n{joined}\n\nORIGINAL:\n{abstract_clean}"
    patched = prompt_model(context3, prompt3)

    # Step 4: Minimal cleanup
    context4 = """
    Proofread the following scientific abstract:
    1. Remove any meta-commentary, preamble, or title (e.g. "Here is the abstract:").
    2. Fix only clear grammatical errors that impede reading.
    3. Do NOT smooth, polish, add transitions, or homogenize the style.
    4. Preserve short sentences and all stylistic irregularities.
    5. Remove any markdown formatting (bold, italics, bullet points) if present.
    Output only the clean abstract text.
    """
    prompt4 = f"Abstract:\n{patched}"
    final = prompt_model(context4, prompt4)

    # Python post-processing: strip any LaTeX that slipped through
    final = re.sub(r'\\[\(\[](.*?)\\[\)\]]', lambda m: m.group(1), final, flags=re.DOTALL)
    final = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', final)

    return final, model_name


def rewrite_function_2(prompt_model, abstract, model_name):
    """t=2 -- The Distiller. Clipped, telegraphic prompt voice.

    Hypothesis: t=1 failures were dominated by long, dense sources (m0_len
    r=-0.46 with Human) that kept their comprehensive essay shape. One hard
    compression pass -- roughly halving length, dissolving contribution lists,
    and deleting the significance closer -- should push those toward the terse,
    bursty profile shared by every t=1 success. The prompt itself is written in
    blunt fragments to see whether a non-fluent prompt yields less fluent output.
    Calls: compress -> fact guard -> cleanup (3).
    """
    abstract_clean = _strip_latex(abstract)

    context1 = """
    Cut this abstract down. Hard.

    Rules. Short ones.
    - Keep every number, dataset, method name, model name, equation, and negation. Exactly.
    - Keep every limitation or caveat that carries real information.
    - Drop filler. Drop throat-clearing. Drop restated motivation.
    - Any list of "contributions" becomes plain running sentences. No labels. No "twofold". No colon then a list.
    - Delete the last sentence if it only sells importance, impact, or future promise. Do not replace it.
    - No markdown. No LaTeX. Math in plain Unicode: write f subset X x Y x Z, not \\(f\\subseteq X\\times Y\\times Z\\).

    Aim for about half the words. Meaning stays. Prose only.
    Return the trimmed abstract. Nothing else.
    """
    prompt1 = f"Abstract:\n{abstract_clean}"
    trimmed = prompt_model(context1, prompt1)

    context2 = """
    Two versions of an abstract: SOURCE and SHORT. SHORT was trimmed from SOURCE.
    Check one thing only -- did the trim change or lose a hard fact?
    - A numeric value in SOURCE now missing or different in SHORT: put the correct value back in the relevant sentence.
    - A named method, dataset, model, equation, or a negation/limitation dropped from SHORT that changes the claim: restore it, briefly.
    - Add nothing SOURCE does not support. Do not re-expand. Do not add a significance sentence.
    - No markdown, no LaTeX.
    Return the corrected SHORT abstract only.
    """
    prompt2 = f"SOURCE:\n{abstract_clean}\n\nSHORT:\n{trimmed}"
    patched = prompt_model(context2, prompt2)

    context3 = """
    Clean pass. Remove any leftover preamble, title, or note like "Here is".
    Fix only grammar that blocks reading. Keep it short and uneven. Do not smooth.
    No markdown, no LaTeX. Return the abstract text only.
    """
    prompt3 = f"Abstract:\n{patched}"
    final = prompt_model(context3, prompt3)

    return _strip_latex_post(final), model_name


def rewrite_function_3(prompt_model, abstract, model_name):
    """t=3 -- The Whiteboard Colleague. Warm, second-person, conversational.

    Hypothesis: t=1 failures over-used first person (we/our 5.8 vs 3.4) in a
    uniform, marketing-flavored register. Reframing the abstract as a researcher
    plainly explaining their own work to a peer -- natural, occasional "we", no
    selling -- may produce the honest asymmetry humans write with. This directly
    tests the meta-hypothesis: a warm, un-templated prompt may beget un-templated
    prose. Calls: retell -> fidelity check -> light cleanup (3).
    """
    abstract_clean = _strip_latex(abstract)

    context1 = """
    Picture yourself at a whiteboard, telling a colleague in your field what you actually did.
    They already know the area, so you don't oversell it -- you just walk them through it, honestly.

    Rewrite the passage below in that voice:
    - Say what the problem was, what you did, and what you found, in whatever order feels natural to explain -- not a fixed template.
    - Use "we" and "our" the way people really do when talking about their own work: sometimes, not in every sentence.
    - Keep all the specifics -- numbers, datasets, method and model names, equations, and anything you're careful NOT to claim. A colleague would notice if you fudged those.
    - Don't market it. No "this is important because", no grand closing line about impact or the future. Just stop when you're done explaining.
    - Plain text. No markdown, no LaTeX -- write math in ordinary Unicode symbols.

    It still has to read as a proper scientific abstract, just an unpolished, human one. Give me only that paragraph.
    """
    prompt1 = f"Passage:\n{abstract_clean}"
    spoken = prompt_model(context1, prompt1)

    context2 = """
    Here are two takes on the same abstract: the ORIGINAL and a RETELLING.
    Skim them side by side and make sure the retelling didn't quietly drop or bend anything that matters --
    a number, a dataset or method or model name, an equation, or a careful qualification the authors clearly wanted to keep.
    If something important slipped, fold it back in with the fewest words possible. Don't add claims the original doesn't make,
    and don't tack on a closing significance line. Plain text, no markdown or LaTeX.
    Give me only the corrected retelling.
    """
    prompt2 = f"ORIGINAL:\n{abstract_clean}\n\nRETELLING:\n{spoken}"
    checked = prompt_model(context2, prompt2)

    context3 = """
    Last look. Strip any lead-in like "Sure, here's..." or a title. Fix a grammar slip only if it trips the reader.
    Leave the casual rhythm and any slightly uneven phrasing alone. No markdown, no LaTeX.
    Return just the abstract.
    """
    prompt3 = f"Abstract:\n{checked}"
    final = prompt_model(context3, prompt3)

    return _strip_latex_post(final), model_name


def rewrite_function_4(prompt_model, abstract, model_name):
    """t=4 -- The Discourse Inverter. Cold, prescriptive, style-manual voice.

    Hypothesis: the surviving AI signal at t=1 was discourse-level, not lexical --
    the canonical motivation->method->results->impact arc, enumerated contribution
    lists, and a significance closer (evalclose r=-0.45). Forcing a non-canonical
    order that opens on a concrete result and forbids the impact closer attacks
    that structure head-on. The clinical numbered-spec prompt is a deliberate tonal
    opposite of t=3. Calls: reorder -> fidelity restore -> cleanup (3).
    """
    abstract_clean = _strip_latex(abstract)

    context1 = """
    Reorganize the following abstract according to the specification below.

    1. Opening. Do not open with background or motivation. Open with a concrete result, object, quantity, or construction from the work.
    2. Order. Present the content in a non-canonical order. The sequence motivation, then method, then results, then significance is disallowed.
    3. Enumerations. Any enumerated or labeled list of contributions must be rewritten as continuous prose. Labels, "firstly/secondly", "twofold", and colon-introduced contribution lists are disallowed.
    4. Closing. The final sentence must convey content. A sentence whose only function is to assert importance, impact, generality, or future promise is disallowed and must be removed.
    5. Rhythm. Sentence and clause lengths must be uneven. Do not normalize them.
    6. Fidelity. All numeric values, datasets, method and model names, equations, negations, and stated limitations are retained without change.
    7. Format. Plain text only. No markdown. No LaTeX; render mathematics in Unicode characters.

    Output: the reorganized abstract, and nothing further.
    """
    prompt1 = f"Abstract:\n{abstract_clean}"
    reordered = prompt_model(context1, prompt1)

    context2 = """
    Two documents follow: REFERENCE and REVISION. REVISION is a reordering of REFERENCE.
    Verify that no quantitative value, dataset, method or model name, equation, negation, or limitation from REFERENCE
    was altered or lost in REVISION. Where one was, reinsert or correct it in place, minimally. Introduce no claim
    absent from REFERENCE. Do not append a significance sentence. Plain text; no markdown or LaTeX.
    Output the corrected REVISION only.
    """
    prompt2 = f"REFERENCE:\n{abstract_clean}\n\nREVISION:\n{reordered}"
    patched = prompt_model(context2, prompt2)

    context3 = """
    Final edit. Delete any preamble, heading, or title. Correct only grammar that impedes comprehension.
    Preserve the reordering and uneven rhythm; do not smooth or homogenize. Remove any markdown or LaTeX.
    Output the abstract text only.
    """
    prompt3 = f"Abstract:\n{patched}"
    final = prompt_model(context3, prompt3)

    return _strip_latex_post(final), model_name


def rewrite_function_5(prompt_model, abstract, model_name):
    """t=5 -- The Cadence Hacker. Playful, game-like prompt.

    Hypothesis: burstiness was the clearest lexical marker of t=1 successes
    (sentence-length std slightly higher in Human texts) and uniform rhythm the
    hallmark of failures. Explicitly engineering jagged sentence lengths, varied
    clause openings, and broken parallelism targets that gap directly, holding the
    facts fixed. Operates on the numbered sentence list (like t=1) but for a
    different purpose -- rhythm, not per-sentence rephrase. Calls: recadence ->
    fact check -> tidy (3).
    """
    abstract_clean = _strip_latex(abstract)
    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))

    context1 = """
    Let's play with rhythm. Machine prose has a tell: every sentence is about the same length, every one opens
    the same way, and they all march in step. We're going to break that march.

    Take the numbered sentences below and rewrite them into one abstract with a deliberately jagged cadence:
    - Make the lengths lurch. A three-word jab. Then a long, winding sentence that carries several linked ideas before it lands. Then something medium. Never two neighbors the same size.
    - Change how sentences begin. Some start with the subject, some with a condition, some with a result, some with a plain verb. No two in a row open the same way.
    - Kill parallelism. If two ideas beg to be written as a matched pair, write them mismatched instead.
    - Keep the facts sacred: every number, dataset, method, model, equation, negation, and caveat survives untouched.
    - Don't end on a victory lap. No sentence whose only job is to say why this matters or what comes next.
    - Plain text, please. No markdown, no LaTeX -- math in ordinary Unicode.

    Hand back only the abstract paragraph.
    """
    prompt1 = f"Sentences:\n{numbered}"
    bursty = prompt_model(context1, prompt1)

    context2 = """
    Quick fact-check between two versions, BEFORE and AFTER. AFTER was rewritten for rhythm; make sure the rhythm
    games didn't drop or distort anything real -- a number, a dataset or method or model name, an equation, a negation,
    or a stated limit. Slip any missing one back in without disturbing the cadence much. Add nothing new. No tacked-on
    significance line. Plain text, no markdown or LaTeX.
    Return only the fixed AFTER.
    """
    prompt2 = f"BEFORE:\n{abstract_clean}\n\nAFTER:\n{bursty}"
    checked = prompt_model(context2, prompt2)

    context3 = """
    Tidy-up pass. Trim any intro line or title. Fix grammar only where a reader would stumble. Keep the jagged rhythm
    and varied openings exactly as they are -- that's the point, don't iron it flat. Strip any markdown or LaTeX.
    Return the abstract only.
    """
    prompt3 = f"Abstract:\n{checked}"
    final = prompt_model(context3, prompt3)

    return _strip_latex_post(final), model_name


def rewrite_function_6(prompt_model, abstract, model_name):
    """t=6 -- The Self-Diagnosing Editor. Reflective, Socratic prompt.

    Hypothesis: rather than impose one fixed transform, have the model first locate
    the specific spots that read as templated -- hollow significance claims, uniform
    rhythm, stacked hedges, labeled contribution lists -- then repair only those.
    This is the union of the failure signals from t=1 turned into a self-critique,
    and tests whether asking the model to reason about flat prose (a very "meta"
    prompt) helps or, per the meta-hypothesis, backfires into more LLM-y output.
    Calls: diagnose -> repair -> fidelity -> cleanup (4).
    """
    abstract_clean = _strip_latex(abstract)

    context1 = """
    Read the passage below the way a sharp editor would, and think about where the writing goes slack.
    Which sentences feel templated rather than said by a person? Look especially for: a significance claim that
    asserts importance without adding information; a run of sentences with the same shape and length; stacked hedges
    that could be one; a list of "contributions" dressed up with labels; a closing line that gestures at impact or
    the future without content.

    Do not rewrite anything yet. Just list, briefly, the specific spots that read as generic or hollow, and say in a
    few words what is wrong with each.
    """
    prompt1 = f"Passage:\n{abstract_clean}"
    critique = prompt_model(context1, prompt1)

    context2 = """
    You have a passage and a short diagnosis of its weak spots. Rewrite the passage so those weak spots are gone:
    - Replace hollow significance claims with a concrete detail, or cut them.
    - Break up any stretch of same-shaped sentences; let the lengths vary honestly.
    - Turn any labeled "contributions" list into ordinary prose.
    - If the ending only gestures at importance, end earlier, on something concrete.
    Leave the parts that already read naturally alone. Keep every number, dataset, method, model, equation, negation,
    and caveat exactly. Don't add claims the passage doesn't make. Plain text -- no markdown, no LaTeX, math in Unicode.
    Return only the rewritten abstract.
    """
    prompt2 = f"Passage:\n{abstract_clean}\n\nDiagnosis:\n{critique}"
    repaired = prompt_model(context2, prompt2)

    context3 = """
    Compare INPUT and OUTPUT. OUTPUT is an edited INPUT. Confirm nothing factual moved: numbers, datasets, method and
    model names, equations, negations, limitations. Restore or correct any that changed, minimally, and add nothing
    INPUT does not support. No new significance sentence. Plain text; no markdown or LaTeX.
    Return the corrected OUTPUT only.
    """
    prompt3 = f"INPUT:\n{abstract_clean}\n\nOUTPUT:\n{repaired}"
    patched = prompt_model(context3, prompt3)

    context4 = """
    Proofread. Remove any preamble, note, or title. Fix only grammar that impedes reading. Do not re-polish or
    re-homogenize the rhythm. Remove any markdown or LaTeX. Output the clean abstract only.
    """
    prompt4 = f"Abstract:\n{patched}"
    final = prompt_model(context4, prompt4)

    return _strip_latex_post(final), model_name


# =============================================================================
# Few-shot human-style arm (t=7..t=11). Each supplies held-out human abstracts as
# style references (never the current row's original). They test whether real
# human writing helps GPT-OSS shed the AI register that description-only prompts
# (t=1..t=6) could not. Fidelity guards compare only against the mirror_0 source,
# and every prompt forbids copying the example text -- examples are for cadence
# and register, not content.
# =============================================================================


def rewrite_function_7(prompt_model, abstract, model_name):
    """t=7 -- Basic 1-shot style emulation. The simplest possible test of the
    few-shot theory: one held-out human abstract, "rewrite mirror_0 in this
    style." No reference machinery. Calls: emulate -> fidelity -> cleanup (3)."""
    abstract_clean = _strip_latex(abstract)
    example = _human_examples(abstract_clean, 1)[0]

    context1 = """
    You are shown one human-written scientific abstract, then a draft abstract.
    Rewrite the draft so it reads as if written by the same person who wrote the human abstract --
    match its rhythm, sentence-length variation, word choice, and how it handles hedging and transitions.
    Hard rules:
    - Keep every fact from the DRAFT unchanged: numbers, datasets, method and model names, equations, negations, limitations.
    - Borrow only style from the human abstract. Do not copy its wording, phrases, or subject matter.
    - No markdown, no LaTeX; math in plain Unicode.
    Return only the rewritten abstract.
    """
    prompt1 = f"HUMAN ABSTRACT (style reference only):\n{example}\n\nDRAFT (rewrite this):\n{abstract_clean}"
    styled = prompt_model(context1, prompt1)

    context2 = """
    SOURCE and REWRITE follow. Confirm the rewrite kept every fact in SOURCE:
    numbers, datasets, method/model names, equations, negations, limitations. Restore or correct any that
    drifted, minimally, and add nothing SOURCE does not state. No markdown, no LaTeX.
    Return only the corrected rewrite.
    """
    prompt2 = f"SOURCE:\n{abstract_clean}\n\nREWRITE:\n{styled}"
    checked = prompt_model(context2, prompt2)

    context3 = """
    Proofread. Remove any preamble or title. Fix only grammar that impedes reading. Do not homogenize the style.
    Remove any markdown or LaTeX. Return the abstract only.
    """
    final = prompt_model(context3, f"Abstract:\n{checked}")
    return _strip_latex_post(final), model_name


def rewrite_function_8(prompt_model, abstract, model_name):
    """t=8 -- 3-shot common-style emulation. Give three held-out human abstracts
    and ask the model to infer the shared human style before rewriting, so it
    triangulates register rather than aping one sample. Calls: emulate -> fidelity
    -> cleanup (3)."""
    abstract_clean = _strip_latex(abstract)
    examples = _human_examples(abstract_clean, 3)

    context1 = """
    Below are three human-written scientific abstracts, then a draft abstract.
    First, notice what the three humans have in common in HOW they write -- not what they write about:
    their sentence-length variety, where they put the main point, how sparingly they hedge, how they open sentences,
    where they are terse and where they let a sentence run. Then rewrite the draft so it shares that human manner.
    Hard rules:
    - Preserve every fact in the DRAFT exactly: numbers, datasets, method and model names, equations, negations, limitations.
    - Take only style from the examples. Never reuse their phrasing or their topics.
    - No markdown, no LaTeX; math in plain Unicode.
    Return only the rewritten abstract -- no notes about the style.
    """
    prompt1 = f"{_format_examples(examples)}\n\nDRAFT (rewrite this):\n{abstract_clean}"
    styled = prompt_model(context1, prompt1)

    context2 = """
    SOURCE and REWRITE follow. Verify the rewrite preserved every fact in SOURCE -- numbers, datasets,
    method/model names, equations, negations, limitations. Fix any drift minimally; add nothing new. No markdown, no LaTeX.
    Return only the corrected rewrite.
    """
    prompt2 = f"SOURCE:\n{abstract_clean}\n\nREWRITE:\n{styled}"
    checked = prompt_model(context2, prompt2)

    context3 = """
    Proofread. Strip any preamble or title, fix only blocking grammar, keep the style and any unevenness.
    Remove markdown or LaTeX. Return the abstract only.
    """
    final = prompt_model(context3, f"Abstract:\n{checked}")
    return _strip_latex_post(final), model_name


def rewrite_function_9(prompt_model, abstract, model_name):
    """t=9 -- Reference pipeline augmented with a few-shot style anchor. This is
    the exact v1 four-step machinery (per-sentence alternating rephrase -> rejoin
    -> factual audit -> cleanup), except the rephrase step is anchored to two
    held-out human abstracts. Tests whether human examples lift the known-good
    reference. Calls: 4 (same as v1)."""
    abstract_clean = _strip_latex(abstract)
    examples = _human_examples(abstract_clean, 2)

    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(sentences))

    context1 = """
    Two human-written abstracts are given first, as a style reference for tone and rhythm only.
    Then a numbered list of sentences from a draft abstract.
    Rephrase each sentence independently -- treat each as a standalone statement -- so that the overall voice
    matches the human style references (their cadence, sentence-length variation, and restraint with hedging).
    Do not use markdown. Do NOT use LaTeX or math delimiters; write math in plain Unicode (e.g., f subset X x Y x Z).

    ALTERNATING LENGTH RULE (strict):
    - Odd sentences (1,3,5,...): a SHORT version, under 10 words, single core fact, no hedging or elaboration.
    - Even sentences (2,4,6,...): a full technical sentence with syntax varied from the original.

    For ALL sentences:
    - At most one 3-word phrase may overlap the original. Do not copy clauses.
    - Preserve all numbers, named methods, datasets, and model names exactly.
    - Use first person ("We", "our") naturally where the original does.
    - Borrow style only from the human references; never reuse their wording or subject matter.
    - Avoid these formulaic phrases: "we propose", "we present", "we introduce", "we demonstrate", "novel",
      "state-of-the-art", "extensive", "robust", "in this paper", "in this work", "furthermore", "moreover",
      "it is worth noting", "leverages", "this approach", "the proposed", "notably", "significantly",
      "outperforms", "facilitate", "enable", "surpasses", "validate", "verify", "highlight", "thus", "thereby",
      "utilize", "employ".
    Output only the numbered list of rephrases, one per line.
    """
    prompt1 = f"STYLE REFERENCES:\n{_format_examples(examples)}\n\nSentences:\n{numbered}"
    rephrased = prompt_model(context1, prompt1)

    context2 = """
    The following is a numbered list of independently written sentences (some short, some long).
    Join them into one coherent abstract paragraph. Make ONLY minimal edits for coherence:
    - fix pronoun/reference ambiguities and boundary grammar;
    - do NOT add transitions, smooth style, or homogenize lengths;
    - preserve every short sentence as-is;
    - no markdown; no LaTeX -- keep math in Unicode.
    Output only the joined abstract.
    """
    joined = prompt_model(context2, f"Sentences:\n{rephrased}")

    context3 = """
    Compare the DRAFT abstract to the ORIGINAL. Correct factual errors ONLY:
    - a wrong numeric value in DRAFT: replace with the correct one;
    - a misidentified method/dataset/model in DRAFT: correct it;
    - if DRAFT merely omits content, do NOT add it; add no new sentences;
    - no structural/section summaries; do not polish; write "use" not "utilize"/"employ";
    - no markdown; no LaTeX.
    If there are no factual errors, return DRAFT unchanged. Output only the abstract.
    """
    patched = prompt_model(context3, f"DRAFT:\n{joined}\n\nORIGINAL:\n{abstract_clean}")

    context4 = """
    Proofread: remove any meta-commentary/preamble/title; fix only clear blocking grammar; do NOT smooth,
    add transitions, or homogenize; preserve short sentences and stylistic irregularities; remove any markdown.
    Output only the clean abstract text.
    """
    final = prompt_model(context4, f"Abstract:\n{patched}")

    final = re.sub(r'\\[\(\[](.*?)\\[\)\]]', lambda m: m.group(1), final, flags=re.DOTALL)
    final = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', final)
    return final, model_name


def rewrite_function_10(prompt_model, abstract, model_name):
    """t=10 -- Distilled style descriptor (indirect few-shot). Pass 1 reads three
    held-out human abstracts and writes a concrete, reusable style profile without
    quoting them; pass 2 rewrites mirror_0 to that profile. Tests whether an
    abstracted style transfers better than raw examples -- a middle ground between
    description-only (t=1..6) and literal few-shot. Calls: profile -> apply ->
    fidelity -> cleanup (4)."""
    abstract_clean = _strip_latex(abstract)
    examples = _human_examples(abstract_clean, 3)

    context1 = """
    Read the human-written abstracts below and write a short, concrete STYLE PROFILE of how they are written:
    typical and range of sentence lengths, how often and where they hedge, first-person usage, how sentences open,
    use of connectives, and where they are blunt versus expansive. Describe patterns only.
    Do NOT quote or paraphrase their content, and do not mention their topics. Output only the profile as a few bullet-free lines.
    """
    profile = prompt_model(context1, _format_examples(examples))

    context2 = """
    You have a STYLE PROFILE describing how certain humans write, and a DRAFT abstract.
    Rewrite the DRAFT so its writing matches the profile. Keep all of the DRAFT's facts intact -- numbers, datasets,
    method and model names, equations, negations, limitations. Change manner, not meaning. Do not add claims.
    No markdown, no LaTeX; math in Unicode. Return only the rewritten abstract.
    """
    styled = prompt_model(context2, f"STYLE PROFILE:\n{profile}\n\nDRAFT:\n{abstract_clean}")

    context3 = """
    SOURCE and REWRITE follow. Confirm every fact in SOURCE survived in REWRITE -- numbers, datasets, method/model
    names, equations, negations, limitations. Restore or correct any drift minimally; add nothing new. No markdown, no LaTeX.
    Return only the corrected rewrite.
    """
    checked = prompt_model(context3, f"SOURCE:\n{abstract_clean}\n\nREWRITE:\n{styled}")

    context4 = """
    Proofread. Remove any preamble or title, fix only blocking grammar, keep the style and unevenness intact.
    Remove markdown or LaTeX. Return the abstract only.
    """
    final = prompt_model(context4, f"Abstract:\n{checked}")
    return _strip_latex_post(final), model_name


def rewrite_function_11(prompt_model, abstract, model_name):
    """t=11 -- Combined best guess. One-shot human style emulation fused with the
    two levers that actually moved the detector earlier: the reference's formulaic-
    phrase bans plus t=4's result-first discourse reordering. Tests whether few-
    shot style compounds with the known-useful structural moves. Calls: rewrite ->
    fidelity -> cleanup (3)."""
    abstract_clean = _strip_latex(abstract)
    example = _human_examples(abstract_clean, 1)[0]

    context1 = """
    You are given one human-written abstract as a style reference, then a draft abstract to rewrite.
    Produce a rewrite that (a) reads in the human reference's voice -- its cadence and sentence-length variation --
    and (b) follows these structural moves:
    - Open on a concrete result, quantity, or object from the draft, NOT on background or motivation.
    - Do not end on a sentence whose only job is to assert importance, impact, or future promise; end on content.
    - Turn any labeled or enumerated "contributions" into ordinary running prose.
    - Vary sentence lengths sharply; avoid parallel matched pairs.
    Constraints:
    - Preserve every fact from the DRAFT: numbers, datasets, method and model names, equations, negations, limitations.
    - Take only style from the human reference; never reuse its wording or subject matter.
    - Avoid these phrases: "we propose", "we present", "we introduce", "we demonstrate", "novel", "state-of-the-art",
      "in this paper", "in this work", "furthermore", "moreover", "leverages", "the proposed", "notably",
      "significantly", "outperforms", "utilize", "employ", "highlight", "thereby".
    - No markdown, no LaTeX; math in plain Unicode.
    Return only the rewritten abstract.
    """
    prompt1 = f"HUMAN ABSTRACT (style reference only):\n{example}\n\nDRAFT (rewrite this):\n{abstract_clean}"
    styled = prompt_model(context1, prompt1)

    context2 = """
    SOURCE and REWRITE follow. Verify every fact in SOURCE survived: numbers, datasets, method/model names,
    equations, negations, limitations. Restore or correct any drift minimally; add nothing SOURCE lacks; do not
    reintroduce a significance closer. No markdown, no LaTeX. Return only the corrected rewrite.
    """
    checked = prompt_model(context2, f"SOURCE:\n{abstract_clean}\n\nREWRITE:\n{styled}")

    context3 = """
    Proofread. Remove any preamble or title, fix only blocking grammar, keep the result-first order and uneven
    rhythm. Remove markdown or LaTeX. Return the abstract only.
    """
    final = prompt_model(context3, f"Abstract:\n{checked}")
    return _strip_latex_post(final), model_name


# =============================================================================
# t=12..t=14 -- sentence-level human anchoring inside v1's atomization. v9 showed
# human exemplars only help when subordinated to the per-sentence rephrase (its 2
# rescues); whole-abstract "rewrite in this style" (t=7,8,10,11) failed at 0/25.
# These push the anchor down to sentence granularity in three ways.
# =============================================================================


def _atomized_tail(prompt_model, abstract_clean, rephrased, model_name):
    """Shared v1 tail: rejoin -> factual audit vs source -> cleanup (3 calls)."""
    joined = prompt_model(_CTX_REJOIN, f"Sentences:\n{rephrased}")
    patched = prompt_model(_CTX_FACTUAL, f"DRAFT:\n{joined}\n\nSOURCE:\n{abstract_clean}")
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}")
    return _strip_latex_post(final), model_name


def rewrite_function_12(prompt_model, abstract, model_name):
    """t=12 -- v1 pipeline, but the rephrase step sees a palette of ~20 held-out
    human SENTENCES for micro-cadence reference (replacing v9's whole-abstract
    anchor). Keeps v1's alternating short/long rule and bans -- purely additive.
    Tests whether sentence-granularity exemplars beat abstract-granularity. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    palette = _human_sentences(abstract_clean, 20)
    palette_txt = "\n".join(f"- {p}" for p in palette)
    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    context1 = f"""
    A palette of human-written sentences is given only as a reference for micro-style: their sentence-length
    variation, how they open, and their unshowy rhythm. Never reuse their wording, phrasing, or subject matter.

    Below the palette is a numbered list of sentences from a draft abstract. Rephrase each independently as a
    standalone statement, letting the human palette guide the cadence.
    ALTERNATING LENGTH RULE (strict): odd sentences (1,3,5,...) a SHORT version under 10 words, single core fact,
    no hedging; even sentences (2,4,6,...) a full technical sentence with syntax varied from the original.
    For ALL: at most one 3-word phrase may overlap the original; preserve all numbers, methods, datasets, model
    names exactly; use first person ("We","our") naturally where the original does. {_BANS_LINE}
    No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rephrases, one per line.
    """
    prompt1 = f"HUMAN SENTENCE PALETTE (style only):\n{palette_txt}\n\nSentences:\n{numbered}"
    rephrased = prompt_model(context1, prompt1)
    return _atomized_tail(prompt_model, abstract_clean, rephrased, model_name)


def rewrite_function_13(prompt_model, abstract, model_name):
    """t=13 -- per-sentence paired cadence templates. Each draft sentence is paired
    with a specific held-out human sentence; the model rephrases the draft's meaning
    to match that human sentence's LENGTH and rhythm (not its content). The human
    cadence replaces v1's rigid odd/even rule, so the burstiness is human-derived
    rather than artificial. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    templates = _human_sentences(abstract_clean, len(sentences))
    paired = "\n".join(
        f"{i + 1}. DRAFT: {s}\n   TEMPLATE (imitate its length and rhythm only): {templates[i]}"
        for i, s in enumerate(sentences)
    )

    context1 = f"""
    Below are numbered draft sentences, each paired with a human-written TEMPLATE sentence. For each item, write a
    new sentence that expresses the DRAFT's meaning but takes its LENGTH and rhythm from the TEMPLATE -- if the
    template is short and blunt, be short and blunt; if it winds, let yours wind. Do NOT use the template's words,
    phrases, or subject matter; it is a cadence guide only.
    Preserve every number, method, dataset, and model name from the DRAFT exactly; keep negations and limitations.
    Use first person ("We","our") naturally where the draft does. {_BANS_LINE}
    No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rewritten sentences, one per line.
    """
    prompt1 = f"Items:\n{paired}"
    rephrased = prompt_model(context1, prompt1)
    return _atomized_tail(prompt_model, abstract_clean, rephrased, model_name)


def rewrite_function_14(prompt_model, abstract, model_name):
    """t=14 -- distilled cadence recipe. Pass 1 reads ~15 held-out human sentences
    and emits a compact cadence recipe (a target length pattern and opening-variety
    rules) without quoting them; pass 2 applies that recipe within the atomized
    rephrase, replacing v1's fixed alternation. Middle ground between literal
    templates (t=13) and a fixed rule. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    bank = _human_sentences(abstract_clean, 15)
    bank_txt = "\n".join(f"- {p}" for p in bank)

    context_recipe = """
    Study the human-written sentences below and distill a short CADENCE RECIPE describing HOW to pace a paragraph
    like these: the spread of sentence lengths (roughly how short the short ones get and how long the long ones run),
    how often lengths change between neighbors, and the variety of sentence openings. Describe patterns as rules a
    writer could follow. Do NOT quote or paraphrase any sentence's content or topic. Output only the recipe, a few lines.
    """
    recipe = prompt_model(context_recipe, bank_txt)

    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    context1 = f"""
    You have a CADENCE RECIPE (below) and a numbered list of draft sentences. Rephrase each sentence independently as
    a standalone statement, pacing the set to follow the recipe -- vary the lengths as it prescribes; do not make the
    sentences uniform. Preserve every number, method, dataset, and model name exactly; keep negations and limitations;
    use first person ("We","our") naturally where the draft does. {_BANS_LINE}
    No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rephrases, one per line.
    """
    prompt1 = f"CADENCE RECIPE:\n{recipe}\n\nSentences:\n{numbered}"
    rephrased = prompt_model(context1, prompt1)

    joined_patched = prompt_model(
        _CTX_JOIN_FACTUAL, f"SENTENCES:\n{rephrased}\n\nSOURCE:\n{abstract_clean}"
    )
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{joined_patched}")
    return _strip_latex_post(final), model_name


# =============================================================================
# t=15..t=16 -- v1's exact pipeline plus result-first / arc-breaking reordering,
# the discourse lever from t=4 (the only move that pulled a stuck high-pai v1 row
# down, 0.82->0.34). v1 rewrites sentences in place and never touches order; these
# add reordering at two different stages. Both stay within the 4-call budget by
# merging v1's rejoin and factual passes.
# =============================================================================


def rewrite_function_15(prompt_model, abstract, model_name):
    """t=15 -- reorder FIRST, then atomize. Pass 1 reorders the source to open on a
    concrete result and dissolve the motivation->method->result->impact arc, dropping
    any significance closer; pass 2 is v1's per-sentence rephrase + bans + alternation
    over the reordered text; pass 3 joins + factual-audits vs the original source;
    pass 4 cleans up. 4 calls."""
    abstract_clean = _strip_latex(abstract)

    context_reorder = """
    Reorganize the abstract below WITHOUT changing any of its facts:
    - Open on a concrete result, quantity, object, or construction from the work -- not on background or motivation.
    - Break the usual motivation-then-method-then-results-then-significance order; present the content in a different order.
    - Turn any labeled or enumerated "contributions" list into ordinary prose.
    - Remove any final sentence whose only job is to assert importance, impact, generality, or future promise.
    - Keep every number, method, dataset, model name, equation, negation, and limitation intact.
    No markdown; no LaTeX (math in Unicode). Output only the reorganized abstract.
    """
    reordered = prompt_model(context_reorder, f"Abstract:\n{abstract_clean}")
    reordered = _strip_latex(reordered)

    sentences = [s.strip() for s in split_into_sentences(reordered) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    context_rephrase = f"""
    Rephrase each numbered sentence below independently as a standalone statement.
    ALTERNATING LENGTH RULE (strict): odd sentences (1,3,5,...) a SHORT version under 10 words, single core fact, no
    hedging; even sentences (2,4,6,...) a full technical sentence with syntax varied from the original. Keep the given
    ORDER of the sentences. For ALL: at most one 3-word phrase may overlap the original; preserve all numbers, methods,
    datasets, and model names exactly; use first person ("We","our") naturally where the sentence does. {_BANS_LINE}
    No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rephrases, one per line.
    """
    rephrased = prompt_model(context_rephrase, f"Sentences:\n{numbered}")

    joined_patched = prompt_model(
        _CTX_JOIN_FACTUAL, f"SENTENCES:\n{rephrased}\n\nSOURCE:\n{abstract_clean}"
    )
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{joined_patched}")
    return _strip_latex_post(final), model_name


def rewrite_function_16(prompt_model, abstract, model_name):
    """t=16 -- reorder DURING the join. v1's per-sentence rephrase runs first
    (preserving atomization/bursts), then the join step both stitches the sentences
    AND reorders to lead on a result, dissolves the arc, and drops the closer; then
    factual audit vs source; then cleanup. Same lever as t=15, applied later so the
    atomized sentences are reordered rather than the raw source. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    context_rephrase = f"""
    Rephrase each numbered sentence below independently as a standalone statement.
    ALTERNATING LENGTH RULE (strict): odd sentences (1,3,5,...) a SHORT version under 10 words, single core fact, no
    hedging; even sentences (2,4,6,...) a full technical sentence with syntax varied from the original. For ALL: at most
    one 3-word phrase may overlap the original; preserve all numbers, methods, datasets, and model names exactly; use
    first person ("We","our") naturally where the sentence does. {_BANS_LINE}
    No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rephrases, one per line.
    """
    rephrased = prompt_model(context_rephrase, f"Sentences:\n{numbered}")

    context_reorder_join = """
    Below is a numbered list of independently rephrased sentences. Assemble them into one abstract paragraph, and while
    doing so reorganize the presentation:
    - Begin the paragraph on a concrete result, quantity, or object -- not on background or motivation.
    - Do not follow the usual motivation-method-results-significance order; reorder the sentences.
    - Fold any enumerated "contributions" into ordinary prose.
    - Drop any sentence whose only job is to assert importance, impact, or future promise.
    Make only minimal coherence edits; do NOT smooth, add transitions, or homogenize sentence lengths; preserve the
    short sentences. Do not add or remove facts. No markdown; no LaTeX (math in Unicode). Output only the paragraph.
    """
    reordered = prompt_model(context_reorder_join, f"Sentences:\n{rephrased}")

    patched = prompt_model(_CTX_FACTUAL, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}")
    return _strip_latex_post(final), model_name


# =============================================================================
# t=17..t=19 -- refinements of v16 (the champion: v1 atomization + reorder-at-join,
# 0.72 Human, strictly dominant over v1). No few-shot. Each targets one objective
# a hand fidelity audit of v16 surfaced: t=17 cut inherited hallucination, t=18
# push the detector score higher, t=19 recover dropped conclusions/comparatives.
# =============================================================================

# Standard v1 atomized per-sentence rephrase (alternating short/long + bans).
_CTX_REPHRASE_ALT = f"""
Rephrase each numbered sentence below independently as a standalone statement.
ALTERNATING LENGTH RULE (strict): odd sentences (1,3,5,...) a SHORT version under 10 words, single core fact, no
hedging; even sentences (2,4,6,...) a full technical sentence with syntax varied from the original. For ALL: at most
one 3-word phrase may overlap the original; preserve all numbers, methods, datasets, and model names exactly; use
first person ("We","our") naturally where the sentence does. {_BANS_LINE}
No markdown; no LaTeX -- math in plain Unicode. Output only the numbered list of rephrases, one per line.
"""

# v16's reorder-at-join step (assemble + lead on a result + dissolve the arc).
_CTX_REORDER_JOIN = """
Below is a numbered list of independently rephrased sentences. Assemble them into one abstract paragraph, and while
doing so reorganize the presentation:
- Begin the paragraph on a concrete result, quantity, or object -- not on background or motivation.
- Do not follow the usual motivation-method-results-significance order; reorder the sentences.
- Fold any enumerated "contributions" into ordinary prose.
- Drop any sentence whose only job is to assert importance, impact, or future promise.
Make only minimal coherence edits; do NOT smooth, add transitions, or homogenize sentence lengths; preserve the
short sentences. Do not add or remove facts. No markdown; no LaTeX (math in Unicode). Output only the paragraph.
"""


def _rephrase_atomized(prompt_model, text):
    sentences = [s.strip() for s in split_into_sentences(text) if s.strip()]
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return prompt_model(_CTX_REPHRASE_ALT, f"Sentences:\n{numbered}")


def rewrite_function_17(prompt_model, abstract, model_name):
    """t=17 -- v16 with a hardened fidelity guard. Same atomize -> reorder-at-join,
    but the audit pass both fixes factual errors AND strips incidental over-specific
    detail (exact hardware capacities, invented section structures, long lists of
    named sub-operators) that inflate fabrication risk, shrinking the surface for
    hallucinations inherited from mirror_0. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    rephrased = _rephrase_atomized(prompt_model, abstract_clean)
    reordered = prompt_model(_CTX_REORDER_JOIN, f"Sentences:\n{rephrased}")

    context_fidelity_hard = """
    Compare DRAFT to SOURCE and do two things:
    1. Correct factual errors: a wrong numeric value, or a misidentified method/dataset/model -> fix to match SOURCE.
       Preserve every negation and limitation.
    2. Trim fabrication risk: remove incidental over-specific details that are NOT part of the core findings, even if
       they look plausible -- exact hardware capacities, enumerated "Section 1/2/3" structures, long lists of named
       sub-operators or parameters, and added quantitative specifics that do not appear in SOURCE. Keep all core
       results, the numbers that ARE findings, methods, datasets, comparisons, negations, and limitations.
    Do NOT add new content or a significance closer. No markdown or LaTeX. Output only the corrected abstract.
    """
    patched = prompt_model(context_fidelity_hard, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}")
    return _strip_latex_post(final), model_name


def rewrite_function_18(prompt_model, abstract, model_name):
    """t=18 -- v16 with maximally aggressive result-first reordering, to test whether
    a harder discourse break pushes past 0.72. Same atomize + factual audit + cleanup;
    the join step opens on the single strongest result and scrambles the canonical
    order as much as coherence allows. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    rephrased = _rephrase_atomized(prompt_model, abstract_clean)

    context_reorder_aggr = """
    Below is a numbered list of independently rephrased sentences. Assemble them into one abstract paragraph and
    restructure it aggressively:
    - Identify the single strongest concrete result or quantity and OPEN the paragraph on it.
    - Do NOT open on background, motivation, or problem setup, and do NOT follow the usual
      motivation-method-results-significance order; present the content in a markedly non-canonical order.
    - Fold any enumerated or labeled contributions into ordinary prose.
    - Remove any sentence whose only role is to assert importance, impact, generality, or future promise.
    - Keep sentence lengths sharply uneven; do not homogenize.
    Make only minimal coherence edits; do not add or remove facts. No markdown or LaTeX. Output only the paragraph.
    """
    reordered = prompt_model(context_reorder_aggr, f"Sentences:\n{rephrased}")
    patched = prompt_model(_CTX_FACTUAL, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}")
    return _strip_latex_post(final), model_name


def rewrite_function_19(prompt_model, abstract, model_name):
    """t=19 -- v16 with an essential-claim restore, addressing the two omission modes
    the audit found: dropped survey/review conclusions and comparative results killed
    by the banned-phrase list. The audit fixes facts AND re-inserts the main takeaway
    and any headline comparison, the latter in neutral (non-promotional) wording. 4 calls."""
    abstract_clean = _strip_latex(abstract)
    rephrased = _rephrase_atomized(prompt_model, abstract_clean)
    reordered = prompt_model(_CTX_REORDER_JOIN, f"Sentences:\n{rephrased}")

    context_restore = """
    Compare DRAFT to SOURCE. Do three things, minimally:
    1. Correct any factual error (wrong number, misidentified method/dataset/model) to match SOURCE.
    2. Ensure the paper's single main takeaway or conclusion from SOURCE is stated in DRAFT; if it was dropped, add it
       back as one plain sentence.
    3. Ensure any headline comparison to prior or existing methods present in SOURCE survives; if it was dropped,
       restore it in neutral wording (e.g., "improves on earlier methods", "lower error than prior approaches") --
       do NOT use promotional words such as "surpasses", "outperforms", "state-of-the-art", "significantly".
    Add nothing SOURCE does not support. Do not add a generic significance/impact closer. Preserve negations and
    limitations. No markdown or LaTeX. Output only the corrected abstract.
    """
    patched = prompt_model(context_restore, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")
    final = prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}")
    return _strip_latex_post(final), model_name


# =============================================================================
# t=20 -- v16 end-to-end, then a verbatim logical reorder. Reproduces the
# results_16 logical-reorder experiment as a single strategy: run v16's full
# pipeline unchanged (atomized rephrase -> result-first reorder-at-join -> factual
# audit -> cleanup), then put the finished sentences into a logical reading order
# WITHOUT changing a single word. Expected to land near the experiment's ~0.68
# Human (below v16's 0.72), modulo a different sample (seed 2043 vs v16's 2039) and
# stochastic generation. Uses 5 prompt_model calls (v16's 4 + the reorder).
# =============================================================================


def _logical_reorder(prompt_model, text):
    """Reorder a text's sentences into a logical reading order, changing NOT ONE
    word (pure permutation). One prompt_model call; falls back to the untouched text
    if the model's ordering is not a clean permutation, so wording can never change."""
    sentences = [s.strip() for s in split_into_sentences(text) if s.strip()]
    if len(sentences) <= 1:
        return text

    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    context = """
    You are given numbered sentences from a scientific abstract, currently in a deliberately scrambled,
    hard-to-follow order. Decide the order that reads most logically -- the natural flow of a scientific
    abstract: background or motivation first, then the problem or gap, then the approach or method, then the
    results, then the conclusion or implication.

    Do NOT rewrite, merge, split, paraphrase, or edit any sentence in any way. Choose only an order.
    Output ONLY the sentence numbers in your chosen order, separated by commas (for example: 4,1,5,2,3).
    Include every number exactly once and output nothing else.
    """
    raw = prompt_model(context, f"Sentences:\n{numbered}")

    n = len(sentences)
    seen, perm = set(), []
    for tok in re.findall(r"\d+", raw):
        i = int(tok)
        if 1 <= i <= n and i not in seen:
            seen.add(i)
            perm.append(i)
    for i in range(1, n + 1):  # append any indices the model dropped, in original order
        if i not in seen:
            perm.append(i)

    reordered = [sentences[i - 1] for i in perm]
    if sorted(reordered) != sorted(sentences):  # never emit altered wording
        return text
    return " ".join(reordered)


def rewrite_function_20(prompt_model, abstract, model_name):
    """t=20 -- v16 unchanged, then a verbatim logical reorder of its output. Steps
    1-4 are exactly v16 (rephrase -> reorder-join -> factual -> cleanup); step 5
    permutes the finished sentences into logical order without changing any wording.
    5 prompt_model calls."""
    abstract_clean = _strip_latex(abstract)

    # --- v16 body (identical to rewrite_function_16) ---
    rephrased = _rephrase_atomized(prompt_model, abstract_clean)                       # 1
    reordered = prompt_model(_CTX_REORDER_JOIN, f"Sentences:\n{rephrased}")            # 2
    patched = prompt_model(_CTX_FACTUAL, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")  # 3
    v16_final = _strip_latex_post(prompt_model(_CTX_CLEANUP, f"Abstract:\n{patched}"))  # 4

    # --- verbatim logical reorder of the finished v16 abstract ---
    return _logical_reorder(prompt_model, v16_final), model_name                       # 5


def rewrite_function_21(prompt_model, abstract, model_name):
    """t=21 -- v20 compressed into 4 calls. Same as v20 (v16 end-to-end then a
    verbatim logical reorder) but v16's separate factual-audit and cleanup passes
    are merged into a single fix-up pass, freeing a call for the reorder within the
    4-call budget: rephrase -> reorder-join -> (factual + cleanup) -> logical reorder.
    Should track v20 closely; any gap isolates the cost of merging those two passes."""
    abstract_clean = _strip_latex(abstract)

    rephrased = _rephrase_atomized(prompt_model, abstract_clean)                       # 1
    reordered = prompt_model(_CTX_REORDER_JOIN, f"Sentences:\n{rephrased}")            # 2

    context_fix = """
    Compare DRAFT to SOURCE and produce a clean final abstract in one pass:
    - Correct factual errors only: a wrong numeric value, or a misidentified method/dataset/model -> fix to match
      SOURCE. Do NOT add omitted content or new sentences. Preserve negations and limitations. Write "use", not
      "utilize"/"employ".
    - Remove any meta-commentary, preamble, or title (e.g. "Here is the abstract:").
    - Fix only grammar that impedes reading; do NOT smooth, add transitions, or homogenize sentence lengths;
      preserve short sentences and stylistic irregularities.
    - Remove any markdown or LaTeX; keep math in plain Unicode.
    Output only the clean abstract text.
    """
    v16_final = _strip_latex_post(
        prompt_model(context_fix, f"DRAFT:\n{reordered}\n\nSOURCE:\n{abstract_clean}")  # 3
    )

    return _logical_reorder(prompt_model, v16_final), model_name                       # 4


# Point this at the next timestep to run. inner_loop.py accepts --timestep t
# (selects rewrite_function_{t}, derives seed = 2023 + t) and otherwise falls back
# to these defaults. t=2..t=19 are completed; the next unused timesteps are 20, 21.
CURRENT_STRATEGY = rewrite_function_20
CURRENT_TIMESTEP = 20
