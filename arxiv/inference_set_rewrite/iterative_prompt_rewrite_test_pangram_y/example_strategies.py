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
