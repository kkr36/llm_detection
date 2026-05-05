def rewrite_strategy_v11(prompt_model, abstract, model_name):
    """
    Single-call: full empirical ruleset distilled from t=1..10 analysis.
    Every known high-score phrase is banned; every known low-score template is offered.
    Hard ≤18-word sentence limit (split any longer sentence). Tests whether a single
    well-specified prompt with the complete empirical vocabulary beats all multi-step
    approaches when the model has the full picture at once.
    """
    context = (
        "You are rewriting an AI-generated scientific abstract so it reads as natural human "
        "academic prose. Preserve all technical content and keep approximate word count.\n\n"
        "SENTENCE LENGTH: Every sentence must be at most 18 words. Split any longer sentence "
        "into two separate sentences.\n\n"
        "RESULT VERBS — use ONLY these two forms for findings:\n"
        "  '[Method/Acronym] revealed [short finding].'\n"
        "  '[Subject] is found to [verb].' OR '[Subject] are found to be [predicate].'\n"
        "NEVER use: showed, demonstrated, confirmed, uncovers, establishes, suggests that, "
        "indicates that, yields (for results), has emerged as.\n\n"
        "METHOD SENTENCES: '[Acronym] was employed to [short verb phrase].'\n"
        "  OR: 'To [verify/analyze] [noun], [passive subject] was [verb].'\n\n"
        "PROBLEM SENTENCES: 'We consider [dense technical noun phrase].'\n\n"
        "RESULT SCOPING: Where a result holds in a specific regime, condition, or domain, "
        "open with: 'In the [regime/limit], [finding] is found.'\n"
        "  OR: 'In [domain/sector], [subject] is found to [verb].'\n\n"
        "CONSTRAINT SENTENCES: 'In other words, [subject] can only [verb] [bounded result].'\n"
        "  Use 'can only' wherever a result is bounded, conditional, or limited.\n\n"
        "VALIDATION SENTENCES: 'These [X] are consistent with [named experimental method] "
        "observations.'\n\n"
        "NEVER produce:\n"
        "- Ordinal starters: 'First,', 'Second,', 'Third,', 'Finally,'\n"
        "- 'thereby [gerund]', 'thus [gerund]', 'hence [gerund]'\n"
        "- 'Overall,', 'In summary,', 'Taken together,', 'Concretely,'\n"
        "- ', confirming that [clause]' appended to a sentence\n"
        "- ', indicating [clause]' appended to a sentence\n"
        "- ', enabling [clause]' appended to a sentence\n"
        "- 'while simultaneously [gerund]'\n"
        "- 'By treating [X] on an equal footing with [Y]'\n"
        "- '[X] has several notable implications'\n"
        "- 'Section N outlines/reports/describes'\n"
        "- Bold markdown (**word**)\n"
        "- Triplet parallel adjective+noun lists\n"
        "- Any sentence longer than 18 words\n\n"
        "Output only the rewritten abstract."
    )
    prompt = f"Rewrite this abstract:\n\n{abstract}"
    final_mirror = prompt_model(context, prompt)
    return final_mirror, model_name


def rewrite_strategy_v12(prompt_model, abstract, model_name):
    """
    Two-call: named-entity extraction → entity-anchored template generation.
    Hypothesis: the single strongest predictor of a low score is the presence of a named
    entity (acronym, researcher name, experiment name, place, physical constant, number)
    in the sentence. Call 1 extracts every named entity and compresses each claim to one
    terse phrase. Call 2 writes the abstract with the constraint that every sentence must
    contain at least one named entity from the list, using only the empirically validated
    bypass templates.
    """
    context_extract = (
        "From the following scientific abstract, extract two things.\n\n"
        "1. ENTITIES: List every named entity — acronyms (e.g., PCA, RFID, ANN), "
        "researcher surnames (e.g., Ahlswede, Csiszár), experiment/system names, "
        "place names, physical constants, and specific numbers. "
        "One per line, prefixed with 'ENTITY:'.\n\n"
        "2. CLAIMS: List every distinct scientific claim (problem, method, result, "
        "conclusion) as a terse phrase of ≤12 words. "
        "One per line, prefixed with 'CLAIM:'.\n\n"
        "No prose. No markdown. Output only the ENTITY: and CLAIM: lines."
    )
    prompt_extract = f"Extract entities and claims:\n\n{abstract}"
    entities_and_claims = prompt_model(context_extract, prompt_extract)

    context_write = (
        "Write a scientific abstract from the extracted claims below. "
        "Every sentence MUST contain at least one of the listed named entities.\n\n"
        "Use ONLY these sentence templates:\n"
        "  A: 'We consider [technical noun with named entity].'\n"
        "  B: '[ENTITY] was employed to [short verb phrase].'\n"
        "  C: '[ENTITY] revealed [short finding].'\n"
        "  D: '[Subject] is found to [verb].' or '[Subject] are found to be [predicate].'\n"
        "  E: 'In the [regime/limit], [finding involving ENTITY] is found.'\n"
        "  F: 'In [domain/sector], [ENTITY] is found to [verb].'\n"
        "  G: '[Subject] are shown to be [short predicate].'\n"
        "  H: 'In other words, [subject] can only [verb] [bounded result].'\n"
        "  I: 'These [X] are consistent with [ENTITY] observations.'\n"
        "  J: '[X] was proposed to [verb].'\n\n"
        "Rules:\n"
        "- Every sentence must be at most 18 words\n"
        "- Every sentence must use one of the templates A-J\n"
        "- Every sentence must contain at least one listed ENTITY\n"
        "- Do NOT use: showed, demonstrated, confirmed, thereby, thus, Overall, Finally, "
        "bold markdown, triplet lists, appended ', confirming that' or ', indicating'\n"
        "Output only the abstract."
    )
    prompt_write = (
        f"Write an abstract from these extracted entities and claims.\n\n"
        f"{entities_and_claims}"
    )
    final_mirror = prompt_model(context_write, prompt_write)
    return final_mirror, model_name


def rewrite_strategy_v13(prompt_model, abstract, model_name):
    """
    Two-call: maximum compression first, then minimal style overlay.
    Call 1 forces aggressive sentence-level compression: every claim reduced to ≤15 words,
    passive voice only, no connective tissue. This is the most aggressive length-reduction
    approach attempted so far — the output is almost a list of raw facts.
    Call 2 does a single targeted style pass: replaces result verbs with 'revealed'/'is found',
    inserts 'can only' for bounded claims, and prepends 'In the [regime],' where inferable.
    Deliberately avoids a reconstruction step; the compression output drives the structure.
    """
    context_compress = (
        "Compress the following scientific abstract into a numbered list of short declarative "
        "sentences. Strict rules:\n"
        "- One sentence per distinct claim\n"
        "- Each sentence MUST be ≤15 words — cut ruthlessly\n"
        "- Passive voice only: '[X] was [verb]', '[X] are [predicate]', '[X] were [verb]'\n"
        "- No connective words between sentences: no 'therefore', 'thus', 'thereby', "
        "'hence', 'consequently', 'moreover'\n"
        "- Replace all result verbs with 'was found to [verb]' or 'were found to be [adj]'\n"
        "- Replace all method verbs with 'was employed to [verb]' or 'was applied to [noun]'\n"
        "- Include every named entity (acronym, name, number) from the original\n"
        "- No markdown, no bold\n"
        "Output only the numbered list."
    )
    prompt_compress = f"Compress to short passive sentences:\n\n{abstract}"
    compressed = prompt_model(context_compress, prompt_compress)

    context_style = (
        "Apply a targeted style pass to the following numbered list of scientific sentences "
        "and assemble them into a coherent abstract. Make only these specific changes:\n\n"
        "1. VERB SUBSTITUTION: Replace any remaining 'showed', 'demonstrated', 'confirmed', "
        "'established', 'suggests', 'indicates' with 'revealed' (for acronym/method subjects) "
        "or 'is found to' (for passive subjects).\n\n"
        "2. CONSTRAINT INSERTION: For result sentences stating a bounded or conditional finding "
        "(X improves Y, X achieves Z, X enables W under condition C), insert 'can only' before "
        "the result verb: '[X] can only [verb] [result]'.\n\n"
        "3. REGIME SCOPING: For result sentences where a specific regime, limit, sector, or "
        "condition is mentioned, restructure as: 'In the [regime/limit], [finding] is found.' "
        "OR 'In [sector], [subject] is found to [verb].'\n\n"
        "4. LENGTH CHECK: Any sentence that is still >18 words after the above — split it "
        "into two sentences at the main clause boundary.\n\n"
        "Do NOT add new content, new sentences, ordinal markers, or summary statements. "
        "Do NOT use: thereby, thus, Overall, confirming that, indicating, bold markdown.\n"
        "Output only the final abstract."
    )
    prompt_style = f"Apply style pass and assemble into abstract:\n\n{compressed}"
    final_mirror = prompt_model(context_style, prompt_style)
    return final_mirror, model_name


def rewrite_strategy_v14(prompt_model, abstract, model_name):
    """
    Three-call: sentence annotation → regime-scoped draft → trailing-clause pruning.
    Call 1 annotates each sentence with its TYPE, primary NAMED ENTITY, and REGIME/DOMAIN.
    Call 2 generates the abstract using only regime-scoped templates — the scoping opener
    ('In the [regime], [finding] is found') was empirically the most consistent sub-0.05
    pattern and is under-exploited by prior strategies.
    Call 3 prunes trailing elaboration clauses (',  confirming that...', ', enabling...',
    'while simultaneously...') — the specific patterns that convert a good short sentence
    into a near-ceiling one by appending a long dependent clause.
    """
    context_annotate = (
        "Annotate each sentence of the following scientific abstract. "
        "Output one line per sentence in this exact format:\n"
        "N | TYPE | ENTITY | REGIME\n\n"
        "TYPE: problem / method / result / conclusion\n"
        "ENTITY: the primary named entity in the sentence "
        "(an acronym, researcher surname, experiment name, place name, or number); "
        "write NONE if absent\n"
        "REGIME: the specific regime, limit, sector, condition, or domain in which the "
        "result holds (e.g., 'low-interaction limit', 'healthcare sector', "
        "'superposition regime', 'isotropic limit'); write NONE if absent\n\n"
        "No prose. Output only the annotation lines."
    )
    prompt_annotate = f"Annotate each sentence:\n\n{abstract}"
    annotations = prompt_model(context_annotate, prompt_annotate)

    context_draft = (
        "Write a scientific abstract using the annotation below. "
        "Apply these template rules strictly by TYPE:\n\n"
        "problem → 'We consider [dense technical noun phrase containing ENTITY].'\n"
        "  (≤18 words; ENTITY must appear)\n\n"
        "method → '[ENTITY] was employed to [short verb phrase].' "
        "  OR: 'To [verify/analyze] [noun], [passive subject] was [verb].'\n"
        "  (≤18 words)\n\n"
        "result (REGIME ≠ NONE) → 'In the [REGIME], [finding involving ENTITY] is found.'\n"
        "result (REGIME = NONE) → '[ENTITY] revealed [short finding].'\n"
        "  OR: '[Subject] is found to [verb].'\n"
        "  (≤18 words)\n\n"
        "conclusion → 'In other words, [subject] can only [verb] [bounded result].'\n"
        "  OR: 'These [X] are consistent with [ENTITY] observations.'\n"
        "  (≤18 words)\n\n"
        "NEVER use: showed, demonstrated, confirmed, thereby, thus, Overall, Finally, "
        "bold markdown, ', confirming that', ', indicating', 'while simultaneously', "
        "any sentence >18 words.\n"
        "Output only the abstract."
    )
    prompt_draft = (
        f"Write an abstract from these annotations.\n\n"
        f"ANNOTATIONS:\n{annotations}\n\n"
        f"ORIGINAL ABSTRACT (for technical content reference):\n{abstract}"
    )
    draft = prompt_model(context_draft, prompt_draft)

    context_prune = (
        "Prune trailing elaboration clauses from the following scientific abstract. "
        "For each sentence, remove any appended clause that matches these patterns:\n"
        "- ', confirming that [rest of sentence]'\n"
        "- ', indicating [rest of sentence]'\n"
        "- ', enabling [rest of sentence]'\n"
        "- ', providing [rest of sentence]'\n"
        "- ', demonstrating [rest of sentence]'\n"
        "- ', suggesting [rest of sentence]'\n"
        "- 'while simultaneously [gerund phrase]'\n"
        "- '—[long elaborating clause]' (em-dash followed by >8 words)\n"
        "- ', which [long relative clause of >8 words]'\n\n"
        "Keep only the main clause of each sentence. "
        "If a sentence has none of these patterns, leave it exactly as written. "
        "Do not add, merge, or rephrase any sentence. "
        "Output only the pruned abstract."
    )
    prompt_prune = f"Prune trailing clauses from this abstract:\n\n{draft}"
    final_mirror = prompt_model(context_prune, prompt_prune)
    return final_mirror, model_name


def rewrite_strategy_v15(prompt_model, abstract, model_name):
    """
    Four-call: compress → draft → classify risk → selective rewrite.
    The key innovation: call 3 classifies every sentence as HIGH_RISK or LOW_RISK based
    on the empirical banlist. Call 4 rewrites ONLY the HIGH_RISK sentences in-place,
    leaving LOW_RISK sentences untouched. This avoids the failure mode of prior correction
    passes that damaged already-good sentences. The draft from call 2 is the floor; calls
    3-4 only improve it, never degrade already-passing sentences.
    """
    context_compress = (
        "Compress the following scientific abstract into a numbered list of short claims. "
        "Each claim:\n"
        "- ≤15 words\n"
        "- One distinct idea (problem / method / result / conclusion)\n"
        "- Include all named entities (acronyms, names, numbers, experiment names)\n"
        "- Use passive voice: 'was [verb]', 'are found', 'were presented', 'was employed'\n"
        "- No connective phrases (no thereby, thus, hence, consequently)\n"
        "No prose. Output only the numbered list."
    )
    prompt_compress = f"Compress:\n\n{abstract}"
    compressed = prompt_model(context_compress, prompt_compress)

    context_draft = (
        "Write a scientific abstract from the following compressed claims. "
        "Apply these templates by claim type:\n"
        "  problem → 'We consider [technical noun phrase].'\n"
        "  method → '[Acronym] was employed to [verb].'\n"
        "  result → '[Acronym] revealed [finding].' OR '[Subject] is found to [verb].'\n"
        "  result with regime → 'In the [regime], [finding] is found.'\n"
        "  result with sector → 'In [sector], [subject] is found to [verb].'\n"
        "  bounded result → '[Subject] can only [verb] [result].'\n"
        "  conclusion → 'In other words, [subject] can only [verb] [result].'\n"
        "  validation → 'These [X] are consistent with [named method] observations.'\n\n"
        "Hard rules:\n"
        "- Every sentence ≤18 words\n"
        "- Never use: showed, demonstrated, confirmed, thereby, thus, Overall, Finally, "
        "bold markdown, triplet adjective+noun lists\n"
        "Output only the abstract."
    )
    prompt_draft = f"Write abstract from compressed claims:\n\n{compressed}"
    draft = prompt_model(context_draft, prompt_draft)

    context_classify = (
        "Classify each sentence of the following scientific abstract as HIGH_RISK or LOW_RISK "
        "for AI detection. Output one line per sentence:\n"
        "N | HIGH_RISK | [sentence]\n"
        "N | LOW_RISK | [sentence]\n\n"
        "Mark HIGH_RISK if the sentence contains ANY of:\n"
        "- A result verb from this list: showed, demonstrated, confirmed, uncovers, "
        "establishes, suggests that, indicates that, has emerged\n"
        "- 'thereby [gerund]' or 'thus [gerund]' or 'hence [gerund]'\n"
        "- 'Overall,' or 'In summary,' or 'Taken together,' or 'Concretely,'\n"
        "- ', confirming that' or ', indicating' or ', enabling' appended to the sentence\n"
        "- 'while simultaneously [gerund]'\n"
        "- 'By treating [X] on an equal footing'\n"
        "- 'has several notable implications'\n"
        "- A sentence longer than 18 words\n"
        "- Triplet parallel adjective+noun lists\n"
        "- Bold markdown (**word**)\n\n"
        "Mark LOW_RISK if none of the above apply.\n"
        "Output only the classification lines."
    )
    prompt_classify = f"Classify sentences:\n\n{draft}"
    classification = prompt_model(context_classify, prompt_classify)

    context_repair = (
        "You are given a scientific abstract and a sentence classification. "
        "Rewrite ONLY the HIGH_RISK sentences using the bypass templates below. "
        "Leave ALL LOW_RISK sentences exactly as written.\n\n"
        "Bypass templates (choose the best fit for each HIGH_RISK sentence):\n"
        "  '[Acronym] revealed [short finding].' (≤18 words)\n"
        "  '[Subject] is found to [verb].' (≤18 words)\n"
        "  'In the [regime], [finding] is found.' (≤18 words)\n"
        "  '[Acronym] was employed to [verb phrase].' (≤18 words)\n"
        "  'In other words, [subject] can only [verb] [result].' (≤18 words)\n"
        "  'These [X] are consistent with [named method] observations.' (≤18 words)\n\n"
        "If a HIGH_RISK sentence is >18 words, split it into two ≤18-word sentences "
        "before applying a template.\n\n"
        "Output only the final abstract with all replacements applied."
    )
    prompt_repair = (
        f"Rewrite HIGH_RISK sentences only.\n\n"
        f"ABSTRACT:\n{draft}\n\n"
        f"CLASSIFICATION:\n{classification}"
    )
    final_mirror = prompt_model(context_repair, prompt_repair)
    return final_mirror, model_name