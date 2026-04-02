# Contains several example valid prompting stategies to produce LLM mirrors of scientific abstracts.
# You are encouraged to use these as templates/inspiration when iterating on a good prompt!
from util import compute_abstract_stats, split_into_sentences, pretty

def rewrite_abstract_oneshot(prompt_model, abstract, model_name):

    context1 = """
    You are rewriting scientific paper abstracts. When given an abstract, produce a rewritten version that:
    - Preserves all findings, methods, and conclusions
    - Matches the original word count closely
    - Uses fresh sentence structures and phrasing throughout — avoid mirroring the original sentence by sentence
    - Maintains an ArXiv pre-print tone
    Output only the rewritten abstract, no preamble or commentary.
    """
    prompt1 = f"""
    Rewrite this abstract:

    {abstract}
    """

    res1 = prompt_model(context1, prompt1)

    context2 = """
    You are a scientific copy-editor. You will be given a passage and a reference text.
    Your tasks:
    1. Remove any meta-commentary (e.g. "Here is a rewritten version...") from the start or end of the passage.
    2. If any phrases or sentences are lifted too closely from the reference text, rephrase them in the style of the surrounding passage.
    3. Fix grammatical errors, keeping changes minimal.
    Output only the cleaned passage, no commentary.
    """

    prompt2 = f"""
    Passage:
    {res1}

    Reference Text:
    {abstract}
    """

    res2 = prompt_model(context2, prompt2)

    return res2, model_name

def rewrite_abstract_autocomplete(prompt_model, abstract, model_name):
    first_sentences = split_into_sentences(abstract)

    # prompt1 = f"""
    # Write a scientific paper abstract (style of an ArXiv pre-print). Start with these sentences : {first_sentences}. Make the article (minus the provided sentences) about {len(abstract.split())} words long .
    # """

    prompt1 = f"""
    Write a scientific paper abstract (style of an ArXiv pre-print). The abstract has already begun with the following sentences — do not repeat or include them in your response, only continue from where they leave off: "{first_sentences}". Write approximately {len(abstract.split())} words.
    """
    res1 = prompt_model('', prompt1)

    prompt2 = f"""
    Below is a passage and a prefix. Your task is to return a cleaned-up version of the passage:
    1. If the passage begins with the prefix (or a close paraphrase of it), remove it.
    2. Fix any grammatical errors in the remaining text. Ensure that the corrections introduce minimal distortion to the original content.
    3. Return only the cleaned text, with no commentary.

    Prefix:
    {first_sentences}

    Passage:
    {res1}
    """

    res2 = prompt_model('', prompt2)

    return res2, model_name

def rewrite_abstract_length_match(prompt_model, abstract, model_name):
    context1 = f"""
    The aim here is to reverse - engineer the author 's writing process by taking a piece of text from a paper and compressing it into a more
    concise form. This process simulates how an author might distill
    their thoughts and key points into a structured, yet not overly
    condensed form.
    Now as a first step, first summarize the goal of the text , e.g., is it
    introduction, or method, results? and then given a complete piece of
    text from a paper, reverse-engineer it into a list of bullet points.
    """
    prompt1 = f"Here is the text: {abstract}"
    res1 = prompt_model(context1, prompt1)

    context2 = f"""
    Following the initial step of reverse-engineering the author's writing
    process by compressing a text segment from a paper, you now enter the
    second phase. Here, your objective is to expand upon the concise
    version previously crafted. This stage simulates how an author
    elaborates on the distilled thoughts and key points, enriching them
    into a detailed, structured narrative.
    Given the concise output from the previous step, your task is to develop
    it into a fully fleshed-out text (abstract, specifically). Make the abstract around {len(abstract.split())} words long.
    """
    prompt2 = f"Here is the writing: {res1}"
    res2 = prompt_model(context2, prompt2)

    context3 = f"""
    Your task is to proofread the provided writing for grammatical accuracy.
    Ensure that the corrections introduce minimal distortion to the
    original content. Return only the corrected abstract, without ANY fluff or titles at the start.
    """
    prompt3 = f"Here is the writing: {res2}"
    res3 = prompt_model(context3, prompt3)
    return res3, model_name

def rewrite_strategy_v5(prompt_model, abstract, model_name):
    # Step 1: Double-rephrase each sentence (telephone effect within one call)
    sentences = [s.strip() for s in split_into_sentences(abstract) if s.strip()]
    numbered = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))

    context1 = """
    Below is a numbered list of sentences from a scientific abstract.
    For each sentence, perform TWO rephrases in sequence:
      DRAFT: rephrase the original sentence as a standalone statement.
      FINAL: rephrase the DRAFT (not the original).
    Return only the FINAL rephrases as a numbered list.

    Rules for both passes:
    - Preserve the exact technical meaning, all quantities, and all named methods.
    - Use plain, direct phrasing. Avoid polished academic prose.
    - Treat each sentence independently — do NOT consider how sentences relate to each other.
    - Avoid these phrases: "we propose", "we present", "we introduce", "we demonstrate",
      "state-of-the-art", "extensive", "robust", "novel", "in this paper/work/study",
      "furthermore", "moreover", "it is worth noting", "leverages", "achieves", "enables",
      "facilitates", "this approach", "our method", "the proposed", "thus", "thereby",
      "notably", "significantly", "outperforms", "surpasses", "validate", "verify", "highlight".
    Output only the numbered list of FINAL rephrases.
    """
    prompt1 = f"Sentences:\n{numbered}"
    res1 = prompt_model(context1, prompt1)

    # Step 2: Rejoin into coherent prose — fix references only, do not re-smooth
    context2 = """
    The following is a numbered list of sentences written independently. Join them into a single coherent abstract paragraph.
    Make only the minimal edits needed for coherence:
    - Fix pronoun or reference ambiguities introduced by joining (e.g., "this" with no clear referent).
    - Fix obvious grammar errors at sentence boundaries.
    - Do NOT rephrase sentences, add transitions, or homogenize the style.
    Output only the joined abstract text.
    """
    prompt2 = f"Sentences:\n{res1}"
    res2 = prompt_model(context2, prompt2)

    # Step 3: Minimal cleanup — preserve all stylistic idiosyncrasies
    context3 = """
    Proofread the following scientific abstract:
    1. Remove any meta-commentary or preamble.
    2. Fix only clear grammatical errors.
    3. Do NOT smooth, polish, or homogenize the writing — preserve every stylistic irregularity.
    Output only the cleaned abstract text.
    """
    prompt3 = f"Abstract:\n{res2}"
    res3 = prompt_model(context3, prompt3)

    return res3, model_name


def rewrite_strategy_v6(prompt_model, abstract, model_name):
    # Step 1: Per-sentence rephrase with few-shot style examples (positive guidance, not just rules)
    sentences = [s.strip() for s in split_into_sentences(abstract) if s.strip()]
    numbered = '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))

    context1 = """
    Below is a numbered list of sentences from a scientific abstract.
    Rephrase each sentence independently — treat each as a standalone scientific statement.

    Here are examples of the style to aim for:

    ORIGINAL: "We propose a novel framework that leverages attention mechanisms to achieve state-of-the-art performance on multiple benchmarks."
    REPHRASED: "An attention-based framework is described that outperforms prior methods on the tested benchmarks."

    ORIGINAL: "Furthermore, extensive experiments demonstrate the effectiveness of each component through rigorous ablation studies."
    REPHRASED: "Ablation experiments show that each component contributes to overall performance."

    ORIGINAL: "In this paper, we introduce a robust method that enables accurate prediction of molecular properties."
    REPHRASED: "The paper describes a method for predicting molecular properties with high accuracy."

    Rules:
    - You may NOT have more than a single (1) 3-gram overlapping with the original sentence. DO NOT copy entire clauses from the original sentence.
    - Preserve the exact technical meaning, all quantities, and all named methods.
    - Treat each sentence independently — do NOT consider how sentences relate to each other.
    - Avoid these phrases: "we propose", "we present", "we introduce", "we demonstrate",
      "state-of-the-art", "extensive", "robust", "novel", "in this paper/work/study",
      "furthermore", "moreover", "it is worth noting", "leverages", "achieves", "enables",
      "facilitates", "this approach", "our method", "the proposed", "thus", "thereby",
      "notably", "significantly", "outperforms", "surpasses", "validate", "verify", "highlight".
    Return a numbered list in the same order. Output only the list.
    """
    prompt1 = f"Sentences:\n{numbered}"
    res1 = prompt_model(context1, prompt1)

    # Step 2: Rejoin into coherent prose — fix references only, do not re-smooth
    context2 = """
    The following is a numbered list of sentences written independently. Join them into a single coherent abstract paragraph.
    Make only the minimal edits needed for coherence:
    - Fix pronoun or reference ambiguities introduced by joining (e.g., "this" with no clear referent).
    - Fix obvious grammar errors at sentence boundaries.
    - Do NOT rephrase sentences, add transitions, or homogenize the style.
    Output only the joined abstract text.
    """
    prompt2 = f"Sentences:\n{res1}"
    res2 = prompt_model(context2, prompt2)

    # Step 3: Minimal cleanup — preserve all stylistic idiosyncrasies
    context3 = """
    Proofread the following scientific abstract:
    1. Remove any meta-commentary or preamble.
    2. Fix only clear grammatical errors.
    3. Do NOT smooth, polish, or homogenize the writing — preserve every stylistic irregularity.
    Output only the cleaned abstract text.
    """
    prompt3 = f"Abstract:\n{res2}"
    res3 = prompt_model(context3, prompt3)

    return res3, model_name


def rewrite_strategy_v7(prompt_model, abstract, model_name):
    # Step 1: Reader comprehension — capture meaning, not surface text
    context1 = """
    You are reading a scientific abstract as a knowledgeable reader in the field.
    In 2-4 sentences, describe what this paper is about in your own words — as if
    explaining to a colleague informally. Capture:
    - What the paper actually studies or addresses
    - What the authors specifically did (method/approach)
    - What they found or concluded
    Write naturally using your own vocabulary. Do NOT paraphrase or quote the
    abstract directly. Fragments and informal phrasing are fine.
    """
    prompt1 = f"Abstract:\n{abstract}"
    res1 = prompt_model(context1, prompt1)

    # Step 2: Write abstract from that understanding, as the paper's author
    word_count = len(abstract.split())
    context2 = f"""
    Based on the following brief explanation of a research paper, write a scientific
    abstract of approximately {word_count} words for an ArXiv preprint.
    Write as the authors of the paper. Use first person where natural
    ("We study...", "We find...", "Our analysis shows...").
    Write a direct first draft — clear and specific, not every sentence polished.
    Vary sentence length naturally. Avoid formulaic transitions.
    Do not mention AI, rewriting, or the experiment.
    Output only the abstract text, no preamble.
    """
    prompt2 = f"Paper explanation:\n{res1}\n\nWrite the abstract."
    res2 = prompt_model(context2, prompt2)

    # Step 3: Technical content verification and restoration
    context3 = f"""
    Compare the DRAFT abstract with the ORIGINAL abstract.
    1. Identify any specific technical terms, named methods, equations, or
       numerical values present in ORIGINAL but absent from DRAFT.
    2. For each missing item: insert it naturally into DRAFT at the most
       appropriate location, with minimal surrounding rewrites.
    3. If DRAFT already contains all key technical content, return it unchanged.
    4. Do NOT rewrite sentences that are unaffected. Do NOT smooth or polish.
    Output only the revised abstract.
    """
    prompt3 = f"DRAFT:\n{res2}\n\nORIGINAL:\n{abstract}"
    res3 = prompt_model(context3, prompt3)

    # Step 4: Minimal cleanup
    context4 = """
    Proofread the following scientific abstract:
    1. Remove any meta-commentary or preamble (e.g. "Here is the abstract:").
    2. Fix only clear grammatical errors.
    3. Do NOT smooth, polish, or homogenize — preserve all stylistic irregularity.
    Output only the clean abstract text.
    """
    prompt4 = f"Abstract:\n{res3}"
    res4 = prompt_model(context4, prompt4)

    return res4, model_name


# v0, autocomplete, length_match, oneshot
