from util import split_into_sentences
import re


def rewrite_strategy(prompt_model, abstract, model_name):
    abstract_clean = re.sub(r'\\\(', '', abstract)
    abstract_clean = re.sub(r'\\\)', '', abstract_clean)
    abstract_clean = re.sub(r'\\\[', '', abstract_clean)
    abstract_clean = re.sub(r'\\\]', '', abstract_clean)

    sentences = [s.strip() for s in split_into_sentences(abstract_clean) if s.strip()]
    numbered = '\n'.join(f'{i + 1}. {s}' for i, s in enumerate(sentences))

    context1 = f"""
    Below is a numbered list of sentences from a scientific abstract.
    Rephrase each sentence independently. Treat each as a standalone statement.
    Do not use any markdown formatting (no bold, no italics, no bullet points, no headers).
    Do NOT use LaTeX notation or math delimiters (no \\(...\\), no \\[...\\], no \\mathrm{{...}}, no \\subseteq, no \\times).
    Represent mathematical symbols in plain Unicode (e.g., write f⊆X×Y×Z, not \\(f\\subseteq X\\times Y\\times Z\\)).

    ALTERNATING LENGTH RULE:
    - Odd-numbered sentences (1, 3, 5, ...): write a short version under 10 words.
      State only the single core fact or claim. Drop hedging, context, and elaboration.
    - Even-numbered sentences (2, 4, 6, ...): write a full technical sentence.
      Vary the syntactic structure from the original.

    For all sentences:
    - At most 1 three-word phrase may overlap with the original sentence. Do not copy clauses.
    - Preserve all numerical values, named methods, datasets, and model names exactly.
    - Use first-person ("We", "our") naturally where the original does. Scientific action verbs such as
      "We show", "We find", "We use", and "We examine" are encouraged.
    - Avoid only these formulaic phrases: "we propose", "we present", "we introduce",
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

    context2 = """
    The following is a numbered list of sentences written independently, with mixed lengths.
    Join them into a single coherent abstract paragraph.
    Make only the minimal edits needed for coherence:
    - Fix pronoun or reference ambiguities introduced by joining.
    - Fix obvious grammar errors at sentence boundaries.
    - Do not add transitions, smooth style, or homogenize sentence lengths.
    - Preserve every short sentence as-is. Do not expand or merge them.
    - Do not use any markdown formatting.
    - Do not use LaTeX notation or math delimiters. Keep all math in plain Unicode characters.
    Output only the joined abstract text.
    """
    prompt2 = f"Sentences:\n{rephrased}"
    joined = prompt_model(context2, prompt2)

    context3 = f"""
    Compare the DRAFT abstract to the ORIGINAL abstract.
    Your only task is to correct factual errors in DRAFT:
    - If a specific numerical value in DRAFT is wrong, replace it with the correct value.
    - If a named method, dataset, or model in DRAFT is wrong, correct it.
    - If DRAFT is merely incomplete, do not add missing content from ORIGINAL.
    - Do not add new sentences under any circumstances.
    - Do not add structural descriptions, section summaries, or organizational content.
    - Do not smooth, polish, or improve the writing.
    - Do not use LaTeX notation or math delimiters.
    - Do not write "utilize" or "employ". Write "use" instead.
    - Do not use any markdown formatting.
    If there are no factual errors, return the DRAFT unchanged.
    Output only the possibly corrected abstract.
    """
    prompt3 = f"DRAFT:\n{joined}\n\nORIGINAL:\n{abstract_clean}"
    patched = prompt_model(context3, prompt3)

    context4 = """
    Proofread the following scientific abstract:
    1. Remove any meta-commentary, preamble, or title.
    2. Fix only clear grammatical errors that impede reading.
    3. Do not smooth, polish, add transitions, or homogenize the style.
    4. Preserve short sentences and stylistic irregularities.
    5. Remove any markdown formatting if present.
    Output only the clean abstract text.
    """
    prompt4 = f"Abstract:\n{patched}"
    final = prompt_model(context4, prompt4)

    final = re.sub(r'\\[\(\[](.*?)\\[\)\]]', lambda m: m.group(1), final, flags=re.DOTALL)
    final = re.sub(r'\\[a-zA-Z]+\{([^{}]*)\}', r'\1', final)

    return final, model_name


CURRENT_STRATEGY = rewrite_strategy
CURRENT_TIMESTEP = 1
