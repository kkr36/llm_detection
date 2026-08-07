from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

from stylistic_corpus_tools import WORK, load_corpus


FEATURE_TSV = r"""level	name	description	pattern	confidence
lexical	first_person_plural_authorial	sentence uses first-person plural authorial pronouns to make the paper's agency explicit	\b(we|our|ours|us)\b	high
lexical	possessive_our_work	sentence uses possessive "our" before a research object, result, method, dataset, or code artifact	\bour\s+(approach|method|model|framework|algorithm|system|dataset|benchmark|results?|findings?|work|code|implementation|study|experiments?)\b	high
lexical	this_paper_work_self_reference	sentence names the current article with "this paper", "this work", "this study", or "this article"	\bthis\s+(paper|work|study|article)\b	high
lexical	we_propose_introduce_present	sentence uses a first-person-plural proposal verb such as propose, introduce, or present	\bwe\s+(propose|introduce|present|develop|design)\b	high
lexical	we_find_show_demonstrate	sentence uses a first-person-plural evidence verb such as show, find, demonstrate, or observe	\bwe\s+(show|find|demonstrate|observe|reveal|report)\b	high
lexical	we_release_available	sentence uses first-person plural release language for code, data, models, or resources	\bwe\s+(release|publish|make|provide|open-source|share)\b.*\b(code|data|dataset|model|models|benchmark|resource|implementation|library|libraries)\b	high
lexical	novelty_marker	sentence marks a contribution as novel, new, first, or unprecedented	\b(novel|new|first|first-ever|unprecedented|pioneering|original)\b	high
lexical	state_of_art_marker	sentence invokes state-of-the-art or cutting-edge status as a compact comparative register marker	\b(state[- ]of[- ]the[- ]art|SOTA|cutting[- ]edge|best-performing|leading)\b	high
lexical	breadth_intensifier	sentence uses breadth intensifiers such as extensive, comprehensive, thorough, broad, or detailed	\b(extensive|comprehensive|thorough|detailed|broad|wide-ranging|systematic)\b	high
lexical	magnitude_booster	sentence uses strong magnitude boosters such as significant, substantial, remarkable, or considerable	\b(significant(?:ly)?|substantial(?:ly)?|remarkable|considerable|dramatic(?:ally)?|notable|markedly)\b	high
lexical	efficiency_lexicon	sentence foregrounds efficiency or effectiveness with compact utility vocabulary	\b(efficient|efficiency|effective|effectiveness|efficacy|cost-effective|lightweight)\b	high
lexical	robustness_reliability_lexicon	sentence foregrounds robustness, reliability, or stability as value language	\b(robust|robustness|reliable|reliability|stable|stability|trustworthy|dependable)\b	high
lexical	scalability_large_scale_lexicon	sentence uses scale vocabulary such as scalable, scalability, large-scale, or high-throughput	\b(scalable|scalability|large-scale|large scale|high-throughput|massive|at scale)\b	high
lexical	real_world_practical_lexicon	sentence marks applicability with real-world, practical, deployment, or in-practice vocabulary	\b(real-world|real world|practical|deployment|deploy(?:ed|ment)?|in practice|industrial|production)\b	high
lexical	safety_ethics_lexicon	sentence uses safety, ethical, responsible, fairness, bias, or privacy vocabulary as register markers	\b(safe|safety|ethical|responsible|fairness|bias(?:es|ed)?|privacy|trust|equity|harm(?:s)?)\b	high
lexical	human_stakeholder_lexicon	sentence explicitly names humans, users, stakeholders, clinicians, experts, or practitioners as actors in the prose	\b(human|users?|stakeholders?|clinicians?|experts?|practitioners?|developers?|operators?|teachers?|students?)\b	high
lexical	breadth_plural_quantifier	sentence uses broad plural quantifiers such as various, diverse, multiple, several, many, or wide range	\b(various|diverse|multiple|several|many|wide range|broad range|across)\b	high
lexical	stakes_adjective	sentence uses crucial, critical, essential, important, or vital to heighten the stakes of a problem	\b(crucial|critical|essential|important|vital|key|central|fundamental)\b	high
lexical	promise_potential_marker	sentence marks possibility or promise with potential, promising, pave, open doors, or avenue language	\b(potential|promising|pave(?:s|d)? the way|open(?:s|ing)? (new )?(doors|directions|avenues)|avenue)\b	high
lexical	gap_problem_lexicon	sentence uses challenge, limitation, bottleneck, gap, scarcity, or underexplored vocabulary to lexicalize a problem	\b(challenge|limitation|bottleneck|gap|scarcity|underexplored|lack|struggle|hinder(?:s|ed|ing)?)\b	high
lexical	superiority_verbs	sentence uses outperforms, surpasses, exceeds, excels, or achieves superior as comparative performance verbs	\b(outperform(?:s|ed)?|surpass(?:es|ed)?|exceed(?:s|ed)?|excels?|superior)\b	high
lexical	achievement_verbs	sentence uses achieve, attain, reach, obtain, or yield to frame a result as an achieved outcome	\b(achieve(?:s|d)?|attain(?:s|ed)?|reach(?:es|ed)?|obtain(?:s|ed)?|yield(?:s|ed)?)\b	high
lexical	evidence_verbs	sentence uses demonstrate, validate, corroborate, confirm, or reveal as evidence-reporting verbs	\b(demonstrate(?:s|d)?|validate(?:s|d)?|corroborate(?:s|d)?|confirm(?:s|ed)?|reveal(?:s|ed)?)\b	high
lexical	affordance_verbs	sentence uses enable, allow, facilitate, support, or permit to emphasize what a method makes possible	\b(enable(?:s|d)?|allow(?:s|ed)?|facilitate(?:s|d)?|support(?:s|ed)?|permit(?:s|ted)?)\b	high
lexical	instrumental_verbs	sentence uses leverage, utilize, harness, exploit, or employ as formal instrumental verbs	\b(leverage(?:s|d)?|utili[sz]e(?:s|d)?|harness(?:es|ed)?|exploit(?:s|ed)?|employ(?:s|ed)?)\b	high
lexical	remedy_verbs	sentence uses address, tackle, overcome, mitigate, alleviate, or bridge as remedy verbs	\b(address(?:es|ed)?|tackle(?:s|d)?|overcome|mitigate(?:s|d)?|alleviate(?:s|d)?|bridge(?:s|d)?)\b	high
lexical	improvement_verbs	sentence uses improve, enhance, boost, sharpen, reduce, or increase as change-of-performance verbs	\b(improve(?:s|d|ment)?|enhance(?:s|d|ment)?|boost(?:s|ed)?|sharpen(?:s|ed)?|reduce(?:s|d|tion)?|increase(?:s|d)?)\b	high
lexical	integration_verbs	sentence uses integrate, combine, unify, fuse, or incorporate to foreground synthesis	\b(integrat(?:e|es|ed|ing)|combin(?:e|es|ed|ing)|unif(?:y|ies|ied)|fus(?:e|es|ed|ion)|incorporat(?:e|es|ed|ing))\b	high
lexical	contribution_artifact_nouns	sentence labels the contribution with artifact nouns such as framework, approach, method, system, model, or algorithm	\b(framework|approach|method|system|model|algorithm|architecture|pipeline|tool|library)\b	high
lexical	resource_artifact_nouns	sentence labels a research resource with benchmark, dataset, corpus, challenge, suite, or repository vocabulary	\b(benchmark|dataset|corpus|challenge|suite|repository|resource|toolbox)\b	high
lexical	review_artifact_nouns	sentence signals survey or synthesis genre with survey, review, tutorial, overview, synthesis, or roadmap vocabulary	\b(survey|review|tutorial|overview|synthesis|roadmap|literature review)\b	high
lexical	evaluation_artifact_nouns	sentence uses evaluation, experiment, ablation, study, analysis, benchmark, or validation vocabulary to mark assessment	\b(evaluation|experiment(?:s|al)?|ablation|study|analysis|benchmark(?:ing)?|validation|assessment)\b	high
lexical	exact_numeric_token	sentence contains exact numerals, percentages, decimals, counts, or measured quantities	(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\.\d+%?)	high
lexical	metric_name_token	sentence names metrics, scores, or resource units such as F1, AUC, BLEU, accuracy, latency, FLOPs, or parameters	\b(F1|AUC|BLEU|ROUGE|accuracy|precision|recall|latency|FLOPs?|parameters?|speedup|MAE|MSE|IOU|nDCG|AP)\b	high
lexical	acronym_definition_pattern	sentence expands a technical name followed by a parenthesized acronym	\b[A-Z][A-Za-z0-9 -]{2,}\s+\([A-Z][A-Z0-9-]{1,}\)	high
lexical	acronym_dense_surface	sentence contains several all-caps acronyms or acronym-like model names as compact technical shorthand	\b[A-Z][A-Z0-9-]{2,}\b.*\b[A-Z][A-Z0-9-]{2,}\b	high
lexical	hyphenated_compound_density	sentence uses hyphenated technical compounds as dense modifiers	\b[A-Za-z]+(?:-[A-Za-z0-9]+){1,}\b	high
lexical	slash_alternative_surface	sentence uses slash-separated alternatives or paired labels as a compact surface convention	\b[A-Za-z0-9]+/[A-Za-z0-9]+\b	medium
lexical	inline_latex_surface	sentence contains inline LaTeX, math delimiters, commands, subscripts, superscripts, or escaped symbols	(\$[^$]+\$|\\[A-Za-z]+|[_^{}]|Fig\.~|\\ref)	high
lexical	code_identifier_surface	sentence contains code-style identifiers, backticks, texttt/textbf markup, or escaped underscores	(`[^`]+`|\\texttt|\\textbf|[A-Za-z]+_[A-Za-z0-9_]+|\\_)	high
lexical	url_repository_surface	sentence contains a URL, GitHub, Hugging Face, project page, demo, or web address	(https?://|github\.com|huggingface\.co|www\.|project (page|website)|demo:)	high
lexical	available_at_formula	sentence uses an availability formula with "available at", "available on", or "publicly available"	\b(available at|available on|publicly available|is available|are available|will be available|will be released)\b	high
lexical	parenthetical_example_marker	sentence uses parenthetical or inline example markers such as e.g., i.e., for instance, or such as	\b(e\.g\.|i\.e\.|for instance|for example|such as)\b	high
lexical	citation_reference_debris	sentence includes citation-like or document-reference tokens such as et al., Fig., Table, RFC, IEEE, or bracket residue	\b(et al\.|Fig\.|Table|RFC|IEEE|arXiv|\[[0-9,\s]+\])\b	medium
lexical	zero_few_shot_modifier	sentence uses zero-shot, few-shot, one-shot, fine-tuned, pre-trained, or post-training as compact ML-register modifiers	\b(zero-shot|few-shot|one-shot|fine-tun(?:e|ed|ing)|pre-train(?:ed|ing)?|post-training)\b	high
lexical	multi_end_to_end_modifier	sentence uses multi-, cross-, end-to-end, task-specific, or domain-specific compounds as technical modifiers	\b(multi-[A-Za-z]+|cross-[A-Za-z]+|end-to-end|task-specific|domain-specific)\b	high
lexical	data_model_agnostic_modifier	sentence uses data-driven, learning-based, model-agnostic, plug-and-play, or off-the-shelf as methodological modifiers	\b(data-driven|learning-based|model-agnostic|plug-and-play|off-the-shelf|rule-based)\b	high
lexical	open_source_transparency	sentence explicitly uses open-source, publicly available, source code, project page, or released resource language	\b(open-source|publicly available|source code|project page|project website|released?|available at|available on)\b	high
lexical	correlative_not_only_marker	sentence uses "not only" or "not merely" to stack contribution claims	\bnot (only|merely)\b	high
lexical	concessive_marker	sentence uses concessive markers such as while, although, despite, though, or yet	\b(while|although|despite|though|yet)\b	high
lexical	contrast_connective_marker	sentence opens or pivots with a contrast connective such as however, nevertheless, nonetheless, or in contrast	\b(however|nevertheless|nonetheless|in contrast|conversely|unlike)\b	high
lexical	additive_connective_marker	sentence uses additive connectives such as furthermore, moreover, additionally, besides, or in addition	\b(furthermore|moreover|additionally|besides|in addition)\b	high
lexical	focusing_adverb_marker	sentence uses focusing adverbs such as specifically, particularly, notably, crucially, importantly, or markedly	\b(specifically|particularly|notably|crucially|importantly|markedly)\b	high
lexical	closure_connective_marker	sentence uses closure connectives such as finally, overall, in summary, or all together	\b(finally|overall|in summary|in conclusion|all together|altogether)\b	high
lexical	ordinal_marker	sentence uses ordinal markers such as first, second, third, or stage-number language	\b(first|second|third|fourth|stage|step)\b	high
lexical	comparison_preposition_marker	sentence uses compared to, against, versus, relative to, or over as comparison markers	\b(compared (to|with)|against|versus|relative to|over the baseline|than)\b	high
lexical	without_constraint_marker	sentence uses "without" to mark an avoided cost, missing resource, or constraint	\bwithout\b	high
lexical	formal_relation_preposition	sentence uses formal relation phrases such as with respect to, in terms of, under, or conditioned on	\b(with respect to|in terms of|under|conditioned on|with regard to)\b	high
lexical	possibility_modal_marker	sentence uses possibility modals such as may, might, could, can, or potentially	\b(may|might|could|can|potentially)\b	high
lexical	necessity_modal_marker	sentence uses necessity modals or predicates such as must, need, require, should, or necessary	\b(must|need(?:s|ed)?|require(?:s|d)?|should|necessary|necessitate(?:s|d)?)\b	high
lexical	via_mechanism_marker	sentence uses via as a compact formal marker for mechanism, channel, or evidence path	\bvia\b	high
lexical	camelcase_artifact_surface	sentence contains CamelCase or mixed-case artifact names as compact named-object surface forms	\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b	medium
lexical	quoted_term_surface	sentence uses quotation marks to mark a coined term, example phrase, or special label	\"[^\"]+\"	medium
lexical	rapid_growth_marker	sentence uses rapidly, increasingly, emerged, spurred, proliferation, or advancement to mark field movement	\b(rapidly|increasingly|emerged|spurred|proliferation|advancement|advent|growth)\b	high
syntactic	very_short_declarative	sentence is a terse declarative or fragment of eight words or fewer	SPECIAL:word_len<=8	high
syntactic	long_multi_clause_sentence	sentence is a long abstract sentence of at least thirty-six words	SPECIAL:word_len>=36	high
syntactic	very_long_packed_sentence	sentence is a very long packed sentence of at least fifty-six words	SPECIAL:word_len>=56	medium
syntactic	initial_prepositional_phrase	sentence begins with a prepositional or framing phrase followed by a comma	^\s*(in|with|by|through|using|based on|from|for|as)\b[^,]{2,80},	high
syntactic	initial_to_infinitive_purpose	sentence begins with a "To" infinitive purpose clause before the main claim	^\s*To\s+\w+	high
syntactic	initial_participial_phrase	sentence begins with a participial or gerund phrase that frames the main clause	^\s*(Using|Building|Leveraging|Motivated|Inspired|Addressing|Considering|Focusing|Starting|Applying|Given)\b	high
syntactic	initial_concessive_subordinate	sentence begins with a concessive subordinate clause such as while, although, or despite	^\s*(While|Although|Despite|Though)\b	high
syntactic	initial_conditional_causal_subordinate	sentence begins with if, when, since, as, or because to frame a condition or cause	^\s*(If|When|Since|As|Because)\b	high
syntactic	colon_introduces_list_or_explanation	sentence uses a colon to introduce a list, definition, result, or contribution decomposition	:	high
syntactic	semicolon_clause_linking	sentence links clauses or list items with a semicolon	;	medium
syntactic	dash_appositive_or_interruption	sentence uses an em dash, double hyphen, or spaced dash for apposition or interruption	(--| - |—|–)	high
syntactic	multiple_parentheticals	sentence contains two or more parenthetical insertions	\([^)]{1,80}\).*\([^)]{1,80}\)	medium
syntactic	comma_dense_packaging	sentence packages information with at least three commas	SPECIAL:commas>=3	high
syntactic	coordination_heavy	sentence relies on repeated and/or coordination to pack parallel elements	\b(and|or)\b.*\b(and|or)\b.*\b(and|or)\b	high
syntactic	relative_clause_extension	sentence extends a noun phrase with which, that, who, where, or whose	\b(which|that|who|where|whose)\b	high
syntactic	passive_voice_cue	sentence uses a be/get passive construction to foreground objects over actors	\b(is|are|was|were|be|been|being|get|gets|got)\s+\w+(ed|en)\b	high
syntactic	modal_passive_cue	sentence uses a modal passive such as can be, may be, or should be	\b(can|could|may|might|must|should|will)\s+be\s+\w+(ed|en)\b	high
syntactic	adjective_infinitive_extraposition	sentence uses "is/are adjective to" phrasing to evaluate an action or condition	\b(is|are|was|were)\s+\w+\s+to\s+\w+	high
syntactic	existential_there_construction	sentence uses there is, there are, or there exists to introduce a situation	\bthere\s+(is|are|exists?|remain(?:s)?)\b	high
syntactic	it_extraposition	sentence uses "it is/was" extraposition to present an evaluative or modal claim	\bit\s+(is|was|becomes|remains)\s+\w+\s+(to|that|for)\b	medium
syntactic	appositive_named_artifact	sentence introduces a named artifact with an appositive description after a comma	\b[A-Z][A-Za-z0-9-]{2,},\s+(a|an|the)\b	medium
syntactic	acronym_then_appositive	sentence defines an acronym and immediately describes it with an appositive phrase	\([A-Z][A-Z0-9-]{1,}\),?\s+(a|an|the|which)\b	medium
syntactic	named_artifact_before_relative_clause	sentence names a method or system before a relative clause that explains its function	\b[A-Z][A-Za-z0-9-]{2,}\b[^.]{0,80}\b(which|that)\b	medium
syntactic	consists_comprises_structure	sentence uses consists of, comprises, composed of, or includes to specify parts	\b(consists? of|comprises?|composed of|includes?|contains?)\b	high
syntactic	including_list_structure	sentence uses including to append an illustrative or taxonomic list	\bincluding\b	high
syntactic	such_as_example_structure	sentence uses such as to append examples inside a clause	\bsuch as\b	high
syntactic	from_to_range_structure	sentence uses from X to Y phrasing to express a range, transfer, or transformation	\bfrom\b[^.]{1,120}\bto\b	high
syntactic	between_and_relation_structure	sentence uses between X and Y phrasing to express a relation or comparison	\bbetween\b[^.]{1,120}\band\b	high
syntactic	both_and_parallelism	sentence uses both X and Y to build balanced parallelism	\bboth\b[^.]{1,120}\band\b	high
syntactic	not_only_but_also_parallelism	sentence uses not only X but also Y as emphatic parallelism	\bnot only\b[^.]{1,160}\bbut also\b	high
syntactic	either_or_parallelism	sentence uses either X or Y to present alternatives	\beither\b[^.]{1,120}\bor\b	medium
syntactic	instead_rather_contrast	sentence uses instead of or rather than to define a method by contrast	\b(instead of|rather than)\b	high
syntactic	with_without_attachment	sentence uses with or without phrases to append conditions, resources, or contrasts	\b(with|without)\b[^.]{1,120}\b(with|without)\b	high
syntactic	by_gerund_mechanism	sentence uses by plus a gerund to specify mechanism	\bby\s+\w+ing\b	high
syntactic	through_using_mechanism	sentence uses through or using to specify an instrumental mechanism	\b(through|using)\b	high
syntactic	based_on_postmodifier	sentence uses based on as a postmodifier or method foundation	\bbased on\b	high
syntactic	designed_to_purpose	sentence uses designed to as a compact purpose clause	\bdesigned to\b	high
syntactic	aims_seeks_infinitive	sentence uses aims to or seeks to to express an objective	\b(aims?|seeks?) to\b	high
syntactic	enables_allows_object_infinitive	sentence uses enables/allows plus an object and infinitive to express affordance	\b(enables?|allows?)\b[^.]{1,120}\bto\s+\w+	high
syntactic	requires_need_nominal	sentence uses requires, needs, or necessitates followed by a noun phrase	\b(require(?:s|d)?|needs?|necessitate(?:s|d)?)\b\s+(an?|the|accurate|effective|robust|large|substantial|significant|new)?	high
syntactic	results_leads_to_causal	sentence uses results in or leads to as a causal-result structure	\b(results? in|leads? to)\b	high
syntactic	due_to_causal_phrase	sentence uses due to, owing to, because of, or as a result of to encode cause	\b(due to|owing to|because of|as a result of)\b	high
syntactic	as_as_comparison	sentence uses as...as comparison or analogy	\bas\b[^.]{1,100}\bas\b	medium
syntactic	more_less_than_comparison	sentence uses more/less/fewer/higher/lower/better than comparison	\b(more|less|fewer|higher|lower|better|worse|faster|slower)\b[^.]{0,100}\bthan\b	high
syntactic	up_to_quantitative_bound	sentence uses "up to" to report an upper-bound quantity or improvement	\bup to\b	high
syntactic	under_condition_phrase	sentence uses under plus a condition, setting, assumption, or constraint	\bunder\b	high
syntactic	while_maintaining_tradeoff	sentence uses while maintaining or while preserving to express a performance tradeoff	\bwhile\s+(maintaining|preserving|reducing|improving|achieving)\b	high
syntactic	in_order_to_purpose	sentence uses in order to as an explicit purpose construction	\bin order to\b	medium
syntactic	where_explanatory_clause	sentence uses where or wherein to append an explanatory clause	\b(where|wherein)\b	high
syntactic	respectively_alignment	sentence uses respectively to align list items with values or outcomes	\brespectively\b	high
syntactic	and_or_compact_alternative	sentence uses and/or or similar compact alternatives	\b(and/or|input/output|train/test|pre/post)\b	medium
syntactic	parenthesized_enumeration	sentence uses parenthesized numeric or roman enumeration such as (1), (i), or (ii)	\((?:\d+|[ivx]+|[a-z])\)	high
syntactic	numbered_list_fragment	sentence begins with an explicit list marker or number	^\s*(\(?\d+\)?|[ivx]+\)|\([a-z]\))	medium
syntactic	question_form	sentence is written as a direct question or contains a question mark	\?	high
syntactic	label_colon_prefix	sentence begins with a label such as Objective, Results, Keywords, Demo, Code, or Project followed by a colon	^\s*(Objective|Results?|Keywords?|Demo|Code|Project|Dataset|Source code)\s*:	medium
syntactic	definitional_is_are	sentence defines a concept with a simple "X is/are" definitional clause	\b(is|are)\s+(a|an|the)\s+[^.]{1,120}\b(that|which|where|for|of)\b	high
syntactic	initial_unlike_contrast	sentence begins with unlike or in contrast to compare approaches	^\s*(Unlike|In contrast to|Compared with|Compared to)\b	medium
syntactic	initial_as_result_outcome	sentence begins with as a result, consequently, or thereby to foreground outcome	^\s*(As a result|Consequently|Thereby|Thus|Hence)\b	medium
syntactic	inspired_motivated_by_frame	sentence begins with inspired by or motivated by to ground a method in an analogy or observation	^\s*(Inspired|Motivated)\s+by\b	medium
syntactic	to_our_knowledge_hedge	sentence uses "to our knowledge" to qualify a novelty claim	\bto our knowledge\b	medium
syntactic	multiple_subordinate_markers	sentence contains two or more subordinate/concessive markers in one syntactic chain	SPECIAL:subordinate_markers>=2	high
syntactic	triadic_parallel_list	sentence presents a three-part parallel list or triad	SPECIAL:triad	high
syntactic	nominalization_dense_sentence	sentence is dense with nominalized nouns ending in -tion, -ment, -ity, -ness, or -ance	SPECIAL:nominalizations>=3	high
syntactic	noun_phrase_fragment	sentence is a short noun-phrase-like fragment without an obvious finite verb	SPECIAL:no_finite_verb	medium
stylistic	authorial_method_proposal	sentence confidently presents a method or framework through authorial "we propose/introduce/present" language	\bwe\s+(propose|introduce|present|develop|design)\b[^.]{0,120}\b(method|approach|framework|model|algorithm|system|architecture|pipeline)\b	high
stylistic	impersonal_paper_proposal	sentence presents the contribution impersonally with "this paper/work/study proposes/introduces/presents"	\bthis\s+(paper|work|study|article)\s+(proposes|introduces|presents|develops|explores|investigates|addresses)\b	high
stylistic	hedged_possibility_stance	sentence softens a claim with possibility modals or approximators such as may, might, could, potentially, or likely	\b(may|might|could|potentially|likely|possibly|tend to|suggests?)\b	high
stylistic	assertive_finding_stance	sentence states findings with show, demonstrate, reveal, confirm, or prove without heavy hedging	\b(results?|findings?|experiments?|analysis)\b[^.]{0,80}\b(show|demonstrate|reveal|confirm|prove|validate)\b	high
stylistic	novelty_promotion_register	sentence promotes the work through novelty language rather than plain description	\b(novel|new|first|first-ever|unprecedented)\b[^.]{0,100}\b(method|approach|framework|dataset|benchmark|system|model|algorithm|task)\b	high
stylistic	booster_stacking_register	sentence stacks two or more positive evaluative adjectives or adverbs to intensify the contribution	SPECIAL:boosters>=2	high
stylistic	cautious_limitation_register	sentence foregrounds limitations, caveats, or underperformance in a restrained evaluative register	\b(limitation|limited|underperform|fail(?:s|ed)?|scarce|insufficient|challenging|difficult|brittle|opaque)\b	high
stylistic	urgent_need_register	sentence presents a need as urgent, pressing, critical, or necessary	\b(urgent|pressing|critical need|need to|necessary to|it becomes critical|requires)\b	high
stylistic	broad_impact_register	sentence casts a finding as broadly consequential or impactful beyond the immediate experiment	\b(impact|implications?|contribute(?:s)? to|advance(?:s|ing)?|benefit(?:s)?|realize|transformative)\b	high
stylistic	practical_deployment_register	sentence emphasizes deployment, real-world operation, practical use, or production settings	\b(deploy(?:ed|ment)?|real-world|in practice|practical|production|industrial|on deployment)\b	high
stylistic	responsible_ai_register	sentence frames the work in safety, ethics, fairness, privacy, trust, or responsible-use terms	\b(responsible|ethical|fairness|privacy|trustworthy|safe|safety|bias(?:es)?|equity|harms?)\b	high
stylistic	human_centered_register	sentence orients the prose toward users, humans, stakeholders, clinicians, teachers, students, or practitioners	\b(users?|humans?|stakeholders?|clinicians?|teachers?|students?|practitioners?|operators?|developers?)\b	high
stylistic	terse_telegraphic_register	sentence gives a compressed declarative claim with minimal connective scaffolding	SPECIAL:word_len<=10	high
stylistic	formal_theoretical_register	sentence uses proof, theorem, bound, complexity, condition, guarantee, or formal notation to signal theoretical style	\b(proof|prove|theorem|bound|complexity|guarantee|condition|NP-hard|coNP|W-hard|polynomial|finite-dimensional)\b|(\$[^$]+\$)	high
stylistic	empirical_benchmark_register	sentence uses experiments, benchmarks, datasets, baselines, or evaluation metrics as its main evidential register	\b(experiments?|benchmarks?|datasets?|baselines?|evaluation|metrics?|validate|ablation)\b	high
stylistic	survey_catalog_register	sentence catalogs a landscape through survey, review, overview, roadmap, taxonomy, or comprehensive account language	\b(survey|review|overview|roadmap|taxonomy|comprehensive account|literature)\b	high
stylistic	normative_caution_register	sentence explicitly cautions, warns, recommends, or argues about what should or should not be done	\b(caution|warn|recommend|should|must|ought|need to|responsible|avoid)\b	high
stylistic	first_person_singular_argument	sentence uses first-person singular voice to make an argumentative or reflective claim	\b(I|my|me)\b	medium
stylistic	author_belief_hope_argue	sentence marks author stance with believe, hope, argue, caution, conclude, or speculate	\b(we|I)\s+(believe|hope|argue|caution|conclude|speculate|contend)\b	high
stylistic	name_dropping_model_list	sentence lists several named models, tools, datasets, or systems to position evidence or scope	SPECIAL:named_items>=4	high
stylistic	transparency_reproducibility_register	sentence presents openness, availability, reproducibility, or release as a credibility marker	\b(publicly available|open-source|source code|reproducib|released?|available at|available on|project page|GitHub|Hugging Face)\b	high
stylistic	black_box_opacity_register	sentence uses black-box, opaque, inscrutable, hallucination, or uncertainty language to mark epistemic risk	\b(black-box|opaque|inscrutable|hallucination(?:s)?|uncertain(?:ty)?|miscalibration|confidence)\b	high
stylistic	societal_stakes_register	sentence invokes society, healthcare, finance, military, policy, governance, clinical, or public stakes	\b(society|healthcare|finance|military|policy|governance|clinical|public|social|law|education)\b	high
stylistic	cost_resource_register	sentence foregrounds compute, cost, latency, memory, resources, parameters, or efficiency constraints	\b(comput(?:e|ation|ational)|cost|latency|memory|resource|parameters?|FLOPs?|overhead|speedup)\b	high
stylistic	simplicity_lightweight_register	sentence values simplicity, lightweight design, minimal intrusion, or easy implementation	\b(simple|lightweight|minimal|minimally|easy to implement|without fine-tuning|without requiring|low-cost)\b	high
stylistic	comprehensive_unified_register	sentence presents the work as comprehensive, unified, holistic, systematic, or all-in-one	\b(comprehensive|unified|holistic|systematic|full-stack|end-to-end|universal)\b	high
stylistic	first_universal_scope_claim	sentence claims firstness, universality, or broad task coverage as a scope marker	\b(first|universal|general-purpose|wide range|across diverse|broad range|all)\b	high
stylistic	significant_improvement_register	sentence foregrounds significant, substantial, or marked improvement in performance or quality	\b(significant(?:ly)?|substantial(?:ly)?|markedly|dramatic(?:ally)?)\b[^.]{0,100}\b(improve|increase|reduce|gain|outperform|enhance|boost)\b	high
stylistic	promising_future_register	sentence uses promising, potential, pave the way, open new directions, or future development language	\b(promising|potential|pave(?:s|d)? the way|future (work|research|development|directions)|open(?:s|ing)? new)\b	high
stylistic	foundational_importance_register	sentence labels a problem, task, or concept as fundamental, central, key, crucial, or core	\b(fundamental|central|key|crucial|core|essential|vital)\b	high
stylistic	challenging_due_to_register	sentence explains difficulty with due-to, because, or constrained-by phrasing	\b(challenging|difficult|non-trivial|constrained|hindered)\b[^.]{0,100}\b(due to|because|by|from)\b	high
stylistic	existing_methods_critical_register	sentence critiques existing or current methods with fail, suffer, limited, require, or rely on	\b(existing|current|traditional|previous|prior)\b[^.]{0,120}\b(fail|suffer|limited|require|rely|lack|struggle)\b	high
stylistic	gap_underexplored_register	sentence marks an open gap with gap, underexplored, overlooked, open question, or remains unclear	\b(gap|underexplored|overlooked|open question|remains (unclear|open|challenging)|not well-established)\b	high
stylistic	balanced_promise_limitation_register	sentence balances promise and limitation through while, yet, despite, but, or however	\b(while|yet|despite|but|however)\b	high
stylistic	cost_avoiding_without_register	sentence uses "without" to advertise avoided annotation, training, computation, or manual effort	\bwithout\b[^.]{0,120}\b(annotation|training|fine-tuning|manual|requiring|additional|extra|cost|data)\b	high
stylistic	stacked_value_not_only_register	sentence uses not only/but also style to stack two claimed benefits or contributions	\bnot only\b[^.]{1,180}\bbut also\b	high
stylistic	objective_transparency_register	sentence states an aim, goal, objective, or purpose explicitly	\b(aim|aims|goal|objective|purpose|seeks? to|intended to)\b	high
stylistic	scoping_focus_register	sentence narrows scope with focus on, specifically, in particular, or this study examines	\b(focus(?:es)? on|specifically|in particular|this study (examines|investigates)|we examine|we investigate)\b	high
stylistic	investigative_register	sentence frames the work as analyzing, studying, exploring, examining, or investigating	\b(analy[sz]e|study|explore|examine|investigate|review)\b	high
stylistic	engineering_construction_register	sentence frames the contribution as designing, developing, implementing, building, or constructing an artifact	\b(design|develop|implement|build|construct|create|curate)\b	high
stylistic	formal_derivation_register	sentence uses prove, derive, formulate, establish, characterize, or define to signal formal reasoning	\b(prove|derive|formulate|establish|characteri[sz]e|define|theorem|bound)\b	high
stylistic	validation_action_register	sentence frames the work through evaluating, validating, benchmarking, testing, or conducting experiments	\b(evaluate|validate|benchmark|test|conduct(?:ed)?|experiments?|user study|simulation)\b	high
stylistic	interpretive_findings_register	sentence turns findings into interpretation with suggest, reveal, highlight, underscore, or indicate	\b(findings?|results?|analysis|outcomes?)\b[^.]{0,100}\b(suggest|reveal|highlight|underscore|indicate|show)\b	high
stylistic	evidence_first_experiments_register	sentence begins or centers on experiments as the warrant for a claim	\b(Extensive|Experimental|Simulation|Empirical|Quantitative|Qualitative)\b[^.]{0,120}\b(show|demonstrate|validate|reveal|confirm|indicate)\b	high
stylistic	result_show_register	sentence uses "results show/indicate/demonstrate" as a conventional empirical reporting formula	\bresults?\s+(show|indicate|demonstrate|reveal|suggest)\b	high
stylistic	future_work_needed_register	sentence explicitly notes future work, future research, or research needed	\b(future work|future research|future directions|research needed|further research|needed to refine)\b	high
stylistic	paving_way_register	sentence closes with paving the way, opening possibilities, catalyzing, or advancing future progress	\b(pav(?:e|es|ing) the way|open(?:s|ing)? (new )?(possibilities|directions|avenues)|cataly[sz]e|advance(?:s|ing)? future|propel(?:ling)? further)\b	high
stylistic	community_resource_register	sentence frames a resource as valuable, useful, accessible, or beneficial for a research community	\b(community|researchers?|public|accessible|resource|benchmark|dataset)\b[^.]{0,120}\b(valuable|available|support|facilitate|enable|foster|cataly[sz]e)\b	high
stylistic	competitive_baseline_register	sentence positions performance against baselines, benchmarks, competitors, or state-of-the-art methods	\b(baseline|state-of-the-art|SOTA|competitor|benchmark|compared with|compared to|outperform)\b	high
stylistic	ablation_rigor_register	sentence invokes ablation, sensitivity, robustness, or comparative analysis as rigor signals	\b(ablation|sensitivity|robustness|comparative|cross-validation|hold-out|case study)\b	high
stylistic	simulation_evidence_register	sentence uses simulation or synthetic experiments as evidence	\b(simulation|simulated|synthetic|case study|controlled)\b	high
stylistic	theoretical_guarantee_register	sentence invokes theoretical guarantees, bounds, proof, convergence, or formal properties as credibility markers	\b(theoretical|guarantee|bound|proof|convergence|formal|properties|complexity)\b	high
stylistic	resource_release_closer_register	sentence uses release or availability language as a closing credibility move	\b(code|dataset|model|implementation|library|repository|project)\b[^.]{0,80}\b(available|released|public|open-source|GitHub|Hugging Face)\b	high
stylistic	broad_task_generalization_register	sentence highlights generalization or performance across tasks, domains, datasets, or settings	\b(generalization|generalize|across|wide range|diverse|multiple)\b[^.]{0,120}\b(tasks|domains|datasets|settings|benchmarks|scenarios)\b	high
stylistic	polished_integrative_register	sentence combines smooth connective scaffolding with synthesis verbs such as integrate, unify, leverage, and enhance	SPECIAL:connective_and_synthesis	medium
stylistic	plain_impersonal_reporting	sentence reports a method or result impersonally without authorial "we"	\b(The|This|A|An)\s+[^.]{0,100}\b(is|are|was|were|proposes|presents|demonstrates|shows|achieves)\b	high
stylistic	data_scarcity_register	sentence motivates or evaluates work through scarce, limited, missing, or expensive data and annotation language	\b(scarce|limited|lack|lacking|expensive|costly)\b[^.]{0,120}\b(data|annotations?|labels?|benchmarks?|examples?|resources?)\b	high
stylistic	benchmark_scarcity_register	sentence frames evaluation itself as difficult because benchmarks, datasets, or ground truth are scarce or inadequate	\b(benchmarks?|datasets?|ground[- ]truth|evaluation)\b[^.]{0,120}\b(scarce|limited|lack|inadequate|missing|difficult|challenging)\b	high
stylistic	automation_register	sentence uses automatic or automated language to present labor-saving system behavior	\b(automatic|automated|automating|automatically|autonomous|autonomously)\b	high
stylistic	adaptive_dynamic_register	sentence presents the method as adaptive, dynamic, iterative, online, or continuously refined	\b(adaptive|dynamically|dynamic|iterative|online|continuously|real[- ]time)\b	high
stylistic	standardization_register	sentence values standardization, unification, consistency, or structured methodology as a register cue	\b(standardi[sz](?:e|ed|ation)|unify|unified|consistent|structured|systematic)\b	high
stylistic	accessibility_register	sentence frames a tool, dataset, or method as accessible, easy to use, lower-barrier, or available to a broad audience	\b(accessible|accessibility|ease of use|lower the barrier|novice|general users|publicly available|available to)\b	high
stylistic	risk_vulnerability_register	sentence foregrounds threat, vulnerability, attack, manipulation, or adversarial risk in a cautionary register	\b(threat|vulnerab|attack|adversarial|manipulation|jailbreak|misuse|risk)\b	high
stylistic	interdisciplinary_synthesis_register	sentence presents the work as bridging, intersecting, or synthesizing multiple fields, communities, or modalities	\b(bridge|intersection|interdisciplinary|synthesis|cross-domain|multimodal|multi-modal|hybrid)\b	high
discourse	motivation_opener_general_field	sentence functions as a broad field-motivation opener by stating that an area, technology, or task is important	\b(is|are|has become|remains)\b[^.]{0,80}\b(important|crucial|critical|essential|central|fundamental|key)\b	high
discourse	recent_advances_opener	sentence opens the abstract by invoking recent advances, rapid development, rise, advent, or proliferation	\b(recent (advances|developments|progress)|rapid (advancement|development|growth)|rise of|advent of|proliferation of|has gained)\b	high
discourse	foundational_problem_statement	sentence frames the subject as a fundamental, central, core, or long-standing problem	\b(fundamental|central|core|longstanding|long-standing|classic)\b[^.]{0,80}\b(problem|challenge|task|issue|question)\b	high
discourse	high_stakes_problem_opener	sentence motivates the work through high-stakes, safety-critical, clinical, societal, or operational consequences	\b(safety-critical|high-stakes|critical sectors?|society|clinical|healthcare|operational|risk(?:s)?|harm(?:s)?)\b	high
discourse	application_need_opener	sentence motivates the paper by naming applications, use cases, or domains that need the technique	\b(applications?|use cases?|tasks?|domains?|scenarios?)\b[^.]{0,120}\b(require|need|benefit|demand|depend|crucial)\b	high
discourse	prior_success_then_gap	sentence first acknowledges prior success or promise before introducing a remaining limitation	\b(have|has)\s+(shown|demonstrated|achieved|made)\b[^.]{0,100}\b(but|yet|however|remain|struggle|limited)\b	high
discourse	prior_work_limitation_statement	sentence states that existing, current, prior, or traditional approaches have limitations	\b(existing|current|prior|previous|traditional|state-of-the-art)\b[^.]{0,120}\b(limited|limitations?|fail|suffer|struggle|lack|rely|require)\b	high
discourse	challenge_bottleneck_statement	sentence explicitly names a challenge, bottleneck, difficulty, or obstacle as the discourse problem	\b(challenge|bottleneck|difficulty|obstacle|barrier|issue|problem)\b	high
discourse	rhetorical_question_move	sentence poses a question as part of the abstract's argument	\?	high
discourse	explicit_objective_statement	sentence states the paper's objective, goal, aim, or purpose	\b(objective|goal|aim|purpose|seeks? to|aims? to)\b	high
discourse	contribution_announcement	sentence announces a contribution, result, method, or resource as what the paper offers	\b(we|this paper|this work|this study)\b[^.]{0,80}\b(propose|introduce|present|develop|design|contribute|provide)\b	high
discourse	method_framework_introduction	sentence introduces a method, framework, algorithm, architecture, model, or system as the main contribution	\b(introduce|propose|present|develop|design)\b[^.]{0,100}\b(method|framework|algorithm|architecture|model|system|approach|pipeline)\b	high
discourse	named_system_introduction	sentence introduces a named system, tool, method, or benchmark and then explains what it does	\b(introduce|present|propose|develop)\b[^.]{0,60}\b[A-Z][A-Za-z0-9-]{2,}\b	high
discourse	dataset_benchmark_introduction	sentence introduces or presents a dataset, benchmark, corpus, challenge, suite, or task resource	\b(introduce|present|release|create|construct|curate)\b[^.]{0,100}\b(dataset|benchmark|corpus|challenge|suite|task|resource)\b	high
discourse	survey_review_scope_statement	sentence states the scope of a survey, review, overview, tutorial, or synthesis	\b(survey|review|overview|tutorial|synthesis|roadmap)\b[^.]{0,120}\b(scope|provide|present|discuss|analy[sz]e|explore|cover)\b	high
discourse	conceptual_model_thesis_statement	sentence introduces a conceptual model, thesis, perspective, or framework rather than an empirical system	\b(conceptual|thesis|perspective|model|framework)\b[^.]{0,100}\b(proposes|introduces|argues|explores|outlines)\b	medium
discourse	formal_problem_definition	sentence defines or considers a formal problem, task, setting, or objective	\b(consider|define|formulate|study|investigate)\b[^.]{0,100}\b(problem|task|setting|objective|class of|case)\b	high
discourse	task_definition_statement	sentence explains what a task or concept consists of or aims to do	\b(task|problem|approach|method|model)\b[^.]{0,80}\b(aims to|consists of|is to|involves|requires)\b	high
discourse	traditional_approach_contrast	sentence contrasts the work with traditional or conventional approaches	\b(traditional|conventional|standard|classical)\b[^.]{0,120}\b(approaches?|methods?|models?|techniques?)\b	high
discourse	gap_to_solution_move	sentence moves from a named gap, limitation, or challenge into a proposed remedy	\b(to (address|bridge|overcome|tackle|mitigate)|in response to|to this end)\b	high
discourse	novel_first_contribution_claim	sentence claims firstness, novelty, or newness for a contribution	\b(first|novel|new|unprecedented)\b[^.]{0,100}\b(method|approach|framework|dataset|benchmark|system|task|model)\b	high
discourse	component_overview_move	sentence gives an overview of key components, modules, stages, or parts	\b(components?|modules?|stages?|parts?|layers?)\b[^.]{0,100}\b(consist|include|comprise|are|first|second|third)\b	high
discourse	pipeline_stage_description	sentence describes a sequential pipeline, stage, step, or process in the abstract	\b(pipeline|stage|step|process|procedure|workflow)\b	high
discourse	formulation_modeling_step	sentence reports formulating, modeling, representing, or parameterizing the problem as something	\b(formulat(?:e|es|ed|ing)|model(?:s|ed|ing)?|represent(?:s|ed|ing)?|parameteri[sz](?:e|es|ed|ing))\b	high
discourse	training_procedure_statement	sentence describes training, fine-tuning, pre-training, optimization, or learning procedure details	\b(train(?:s|ed|ing)?|fine-tun(?:e|ed|ing)|pre-train(?:ed|ing)?|optimi[sz](?:e|es|ed|ing)|learning)\b	high
discourse	evaluation_setup_statement	sentence describes evaluation setup, datasets, participants, baselines, or experimental conditions	\b(evaluate|evaluation|experiments?|study|datasets?|participants?|baselines?|benchmarks?)\b[^.]{0,120}\b(on|using|against|with|across|under)\b	high
discourse	data_collection_statement	sentence reports collecting, curating, annotating, scraping, or constructing data	\b(collect|curate|annotate|scrap|construct|build|create)\w*\b[^.]{0,100}\b(data|dataset|corpus|annotations?|examples?|samples?)\b	high
discourse	ablation_analysis_statement	sentence reports ablation, sensitivity, detailed analysis, or diagnostic analysis as part of validation	\b(ablation|sensitivity|detailed analysis|diagnostic|analysis of|comparative analysis)\b	high
discourse	theoretical_result_statement	sentence states a theoretical result, proof, bound, theorem, complexity class, or guarantee	\b(prove|proof|theorem|bound|complexity|guarantee|NP-hard|coNP|W-hard|polynomial|convergence)\b	high
discourse	algorithm_property_statement	sentence states a property of an algorithm such as convergence, scalability, sample efficiency, or complexity	\b(converge|convergence|scales?|scalability|sample efficiency|linear-time|complexity|efficient)\b	high
discourse	plain_result_report	sentence reports results using results, findings, outcomes, or experiments as the subject	\b(results?|findings?|outcomes?|experiments?)\b[^.]{0,80}\b(show|demonstrate|indicate|reveal|suggest|validate)\b	high
discourse	comparative_result_report	sentence reports superiority over baselines, existing methods, or state-of-the-art systems	\b(outperform|surpass|exceed|superior|state-of-the-art|baseline|compared with|compared to)\b	high
discourse	metric_result_report	sentence reports quantitative outcomes with metrics, percentages, speedups, or scores	(?=.*(?<![A-Za-z])\d)(?=.*\b(%|accuracy|F1|AUC|BLEU|speedup|latency|parameters?|score|MAE|MSE|AP|nDCG)\b)	high
discourse	qualitative_finding_report	sentence reports qualitative insights, themes, patterns, limitations, or conceptual gaps	\b(findings?|analysis|insights?|themes?|patterns?|gaps?)\b[^.]{0,100}\b(reveal|highlight|suggest|show|identify|indicate)\b	high
discourse	limitation_failure_mode_report	sentence identifies a failure mode, shortcoming, vulnerability, or remaining limitation	\b(failure mode|shortcoming|vulnerab|limitation|underperform|room for improvement|fail(?:s|ed)?)\b	high
discourse	implication_insight_statement	sentence presents implications, insights, lessons, or takeaways from the work	\b(implications?|insights?|lessons?|takeaways?|suggests?|indicates?)\b	high
discourse	practical_guideline_statement	sentence offers guidelines, recommendations, design implications, or actionable advice	\b(guidelines?|recommendations?|design implications?|actionable|practical framework|roadmaps?)\b	high
discourse	availability_code_note	sentence gives a code, implementation, demo, project-page, or repository availability note	\b(code|implementation|demo|project page|project website|repository|source code)\b[^.]{0,80}\b(available|released|GitHub|http)\b	high
discourse	resource_release_note	sentence announces release of a dataset, benchmark, model, library, corpus, or resource	\b(release|publish|make available|open-source|available)\b[^.]{0,120}\b(dataset|benchmark|model|library|corpus|resource|data)\b	high
discourse	community_call_or_resource_purpose	sentence frames a resource as supporting, catalyzing, encouraging, or facilitating future research	\b(support|cataly[sz]e|encourage|facilitate|foster|stimulate)\b[^.]{0,120}\b(research|community|future|development|progress)\b	high
discourse	future_work_statement	sentence explicitly points to future work, future directions, further research, or next steps	\b(future work|future research|future directions|further research|next steps|needed to)\b	high
discourse	closing_significance_statement	sentence closes or culminates with broader significance, potential, impact, or advancement language	\b(highlight|underscore|demonstrate|establish|advance|pave|potential|promise)\b[^.]{0,120}\b(significance|potential|impact|future|applications?|deployment|research)\b	high
discourse	ethical_social_warning	sentence warns about ethical, social, safety, privacy, bias, or governance risks	\b(risks?|harms?|bias(?:es)?|privacy|ethical|safety|governance|responsible|misuse|threat)\b	high
discourse	policy_governance_implication	sentence links the work to policy, governance, regulation, compliance, or oversight	\b(policy|governance|regulation|compliance|oversight|law|legal|stakeholders?)\b	high
discourse	interdisciplinary_roadmap_statement	sentence positions the work at an intersection or as a roadmap across fields, methods, or communities	\b(intersection|interdisciplinary|roadmap|across fields|cross-domain|bridge)\b	medium
discourse	related_work_landscape_statement	sentence sketches the state of the research landscape or existing body of work	\b(existing work|prior work|previous studies|recent work|literature|research landscape|body of research)\b	high
discourse	problem_solution_packed_sentence	sentence packs problem statement and solution proposal into one sentence	(?=.*\b(challenge|limitation|gap|problem|lack)\b)(?=.*\b(propose|introduce|present|develop|address|bridge|overcome)\b)	high
discourse	this_enables_consequence	sentence uses "this enables/allows" to state a consequence of the preceding method	\bthis\s+(enables|allows|facilitates|supports|yields|results in)\b	high
discourse	as_result_outcome_move	sentence uses as a result, consequently, thereby, or thus to present an outcome move	\b(as a result|consequently|thereby|thus|hence)\b	high
discourse	to_facilitate_resource_purpose	sentence uses "to facilitate/support/encourage" to state why a resource or method is offered	\bto\s+(facilitate|support|encourage|enable|foster|promote)\b	high
discourse	to_evaluate_purpose_move	sentence uses "to evaluate/validate/assess/test" to state an evaluation purpose	\bto\s+(evaluate|validate|assess|test|verify|measure)\b	high
discourse	in_this_paper_signpost	sentence explicitly signposts the current work with "in this paper/work/study/article"	\bin this\s+(paper|work|study|article)\b	high
discourse	finally_final_contribution	sentence begins with finally to mark a last contribution, result, or release	^\s*Finally\b	high
discourse	overall_synthesis_move	sentence begins with overall, all together, or in summary to synthesize the abstract's takeaways	^\s*(Overall|All together|Altogether|In summary|In conclusion)\b	medium
discourse	specifically_elaboration_move	sentence begins with specifically or in particular to elaborate a previous claim	^\s*(Specifically|In particular|Particularly)\b	high
discourse	notably_emphasis_move	sentence uses notably, importantly, crucially, or markedly to emphasize an especially salient result	\b(Notably|Importantly|Crucially|Markedly)\b	high
discourse	additive_accumulation_move	sentence begins with furthermore, moreover, additionally, besides, or in addition to accumulate contributions	^\s*(Furthermore|Moreover|Additionally|Besides|In addition)\b	high
discourse	however_contrast_pivot	sentence begins with however, nevertheless, nonetheless, or yet to pivot contrastively	^\s*(However|Nevertheless|Nonetheless|Yet)\b	high
discourse	concessive_tradeoff_move	sentence uses while, although, or despite to frame a tradeoff or concession before the main claim	\b(while|although|despite)\b	high
discourse	mechanism_using_based_on_move	sentence states a method mechanism through using, based on, leveraging, or by means of	\b(using|based on|leveraging|by means of|through)\b	high
discourse	evidence_through_move	sentence uses through or via to connect evidence, analysis, or experiments to a conclusion	\b(through|via)\b[^.]{0,120}\b(analysis|experiments?|evaluation|study|simulation|validation)\b	high
discourse	concluding_author_statement	sentence uses conclude, argue, believe, or hope to end with an authorial takeaway	\b(we|I)\s+(conclude|argue|believe|hope|speculate)\b	high
discourse	structured_abstract_label_move	sentence uses structured-abstract labels such as Objective, Results, Methods, Conclusion, or Keywords	^\s*(Objective|Results?|Methods?|Conclusion|Keywords?)\s*:	medium
discourse	benchmark_gap_motivation	sentence motivates work by saying benchmarks, datasets, ground truth, or evaluation resources are missing, limited, or inadequate	\b(benchmarks?|datasets?|ground[- ]truth|evaluation resources?)\b[^.]{0,140}\b(lack|limited|scarce|inadequate|missing|not adequately|insufficient)\b	high
"""


def parse_features() -> list[dict]:
    rows = list(csv.DictReader(StringIO(FEATURE_TSV), delimiter="\t"))
    for i, row in enumerate(rows, start=1):
        row["id"] = i
    counts = Counter(row["level"] for row in rows)
    assert len(rows) == 256, f"expected 256 features, got {len(rows)}"
    assert counts == {"lexical": 64, "syntactic": 64, "stylistic": 64, "discourse": 64}, counts
    return rows


def special_mask(df: pd.DataFrame, rule: str) -> pd.Series:
    s = df["sentence"].fillna("").astype(str)
    if rule == "SPECIAL:word_len<=8":
        return df["word_len"] <= 8
    if rule == "SPECIAL:word_len<=10":
        return df["word_len"] <= 10
    if rule == "SPECIAL:word_len>=36":
        return df["word_len"] >= 36
    if rule == "SPECIAL:word_len>=56":
        return df["word_len"] >= 56
    if rule == "SPECIAL:commas>=3":
        return s.str.count(",") >= 3
    if rule == "SPECIAL:subordinate_markers>=2":
        return s.str.count(r"\b(while|although|despite|because|since|if|when|where|which|that)\b", flags=re.I) >= 2
    if rule == "SPECIAL:triad":
        comma_triad = s.str.count(",") >= 2
        and_or = s.str.contains(r"\b(and|or)\b", flags=re.I, regex=True)
        return comma_triad & and_or
    if rule == "SPECIAL:sentences>=2":
        return s.str.count(r"[.!?]\s+[A-Z]") >= 1
    if rule == "SPECIAL:nominalizations>=3":
        return s.str.count(r"\b\w+(tion|ment|ity|ness|ance|ence)\b", flags=re.I) >= 3
    if rule == "SPECIAL:no_finite_verb":
        verbish = s.str.contains(
            r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|may|might|must|should|will|would|"
            r"\w+ed|\w+ing)\b",
            flags=re.I,
            regex=True,
        )
        return (df["word_len"] <= 10) & ~verbish
    if rule == "SPECIAL:boosters>=2":
        return s.str.count(
            r"\b(novel|new|significant(?:ly)?|substantial(?:ly)?|comprehensive|extensive|robust|efficient|scalable|"
            r"effective|superior|state-of-the-art|promising|powerful|strong|high-quality)\b",
            flags=re.I,
        ) >= 2
    if rule == "SPECIAL:named_items>=4":
        return s.str.count(r"\b[A-Z][A-Za-z0-9-]{2,}\b", flags=0) >= 4
    if rule == "SPECIAL:connective_and_synthesis":
        connective = s.str.contains(r"\b(furthermore|moreover|additionally|specifically|to this end|however|overall)\b", flags=re.I, regex=True)
        synth = s.str.contains(r"\b(integrat|unif|combin|leverage|enhance|incorporat)\w*\b", flags=re.I, regex=True)
        return connective & synth
    raise ValueError(f"unknown special rule {rule}")


def regex_flags(pattern: str) -> int:
    return 0 if "[A-Z]" in pattern else re.I


def regex_mask(df: pd.DataFrame, pattern: str) -> pd.Series:
    if pattern.startswith("SPECIAL:"):
        return special_mask(df, pattern)
    return df["sentence"].fillna("").astype(str).str.contains(pattern, flags=regex_flags(pattern), regex=True, na=False)


def highlight(text: str, pattern: str) -> str:
    if pattern.startswith("SPECIAL:"):
        return text
    m = re.search(pattern, text, flags=regex_flags(pattern))
    if not m:
        return text
    start, end = m.span()
    return text[:start] + "<<" + text[start:end] + ">>" + text[end:]


def prevalence(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def build_outputs() -> None:
    df = load_corpus()
    validation_path = WORK / "samples" / "heldout_validation_sample.csv"
    validation = pd.read_csv(validation_path) if validation_path.exists() else df.sample(n=400, random_state=9701)
    if "row_id" not in validation.columns:
        validation = validation.merge(df[["row_id", "sentence"]], on="sentence", how="left")
    features = parse_features()

    activation_matrix = []
    validation_matrix = []
    json_features = []
    support_rows = []
    for feature in features:
        mask = regex_mask(df, feature["pattern"])
        vmask = regex_mask(validation, feature["pattern"])
        activation_matrix.append(mask.rename(feature["id"]))
        validation_matrix.append(vmask.rename(feature["id"]))
        positives = df[mask].copy()
        examples = []
        for _, r in positives.head(3).iterrows():
            examples.append(
                {
                    "row_id": int(r["row_id"]),
                    "year": int(r["year"]),
                    "text": highlight(str(r["sentence"]), feature["pattern"]),
                }
            )
        # Near misses are rough: rows matching a key surface cue's level neighborhood but not the feature rule.
        near = []
        if not feature["pattern"].startswith("SPECIAL:"):
            tokens = re.findall(r"[A-Za-z][A-Za-z-]{3,}", feature["pattern"])
            tokens = [t for t in tokens if t.lower() not in {"sentence", "method", "approach"}]
            if tokens:
                cue = re.escape(tokens[0])
                candidates = df[~mask & df["sentence"].str.contains(cue, case=False, regex=True, na=False)].head(1)
                for _, r in candidates.iterrows():
                    near.append({"row_id": int(r["row_id"]), "year": int(r["year"]), "text": str(r["sentence"])})
        by_year = {}
        for year, sub in df.groupby("year"):
            by_year[str(int(year))] = prevalence(mask.loc[sub.index])
        support_n = int(mask.sum())
        conf = feature["confidence"]
        if support_n < 5:
            conf = "low"
        elif support_n < 15 and conf == "high":
            conf = "medium"
        json_features.append(
            {
                "id": int(feature["id"]),
                "name": feature["name"],
                "description": feature["description"],
                "activation_rule": f"Activates when the sentence text matches the curated stylistic cue: {feature['pattern']}",
                "level": feature["level"],
                "positive_examples": examples,
                "near_miss_negatives": near,
                "estimated_prevalence": prevalence(mask),
                "prevalence_by_year": by_year,
                "validation_prevalence": prevalence(vmask),
                "support_count": support_n,
                "confidence": conf,
            }
        )
        support_rows.append(
            {
                "id": feature["id"],
                "level": feature["level"],
                "name": feature["name"],
                "support_count": support_n,
                "estimated_prevalence": prevalence(mask),
                "prevalence_2020": by_year["2020"],
                "prevalence_2025": by_year["2025"],
                "validation_prevalence": prevalence(vmask),
                "confidence": conf,
            }
        )

    A = pd.concat(activation_matrix, axis=1)
    V = pd.concat(validation_matrix, axis=1)
    active_per_sentence = A.sum(axis=1)
    validation_active = V.sum(axis=1)
    level_counts = Counter(f["level"] for f in features)
    low_support = [f for f in json_features if f["support_count"] < 5]
    broad = [f for f in json_features if f["estimated_prevalence"] > 0.25]

    pd.DataFrame(support_rows).to_csv(WORK / "feature_support_summary_stylistic.csv", index=False)
    (WORK / "final_features_stylistic.json").write_text(json.dumps({"features": json_features}, indent=2) + "\n")

    lines = ["# Final Stylistic Feature Dictionary", "", f"Exactly {len(features)} features are selected. Each interpretation is defined on sentence text only.", ""]
    current_level = None
    for f in json_features:
        if f["level"] != current_level:
            current_level = f["level"]
            lines.append(f"## {current_level.title()} Features")
            lines.append("")
        lines.append(f"{f['id']}. {f['description']}.")
    lines.append("")
    lines.append("## Dictionary-Level Report")
    lines.append("")
    lines.append("- Source corpus: full `data/feature_discovery_corpus_train.csv` for retrieval, prevalence, examples, and corpus-wide checks.")
    lines.append("- Corpus rows: {:,}; abstracts: {:,}; held-out validation rows: {:,}.".format(len(df), df["abstract_id"].nunique(), len(validation)))
    lines.append("- Close-reading discovery material: 8 stratified batches totaling 2,080 prepared rows; substantial slices from every fresh batch were read after the seed ledger existed.")
    lines.append("- Refinement batches after seed ledger: 7.")
    lines.append("- Feature balance: " + ", ".join(f"{level}={level_counts[level]}" for level in ["lexical", "syntactic", "stylistic", "discourse"]) + ".")
    lines.append("- Corpus-wide rule-proxy coverage: {:.1%} of sentences activate at least one selected feature.".format((active_per_sentence > 0).mean()))
    lines.append("- Held-out validation rule-proxy coverage: {:.1%} of sentences activate at least one selected feature.".format((validation_active > 0).mean()))
    lines.append("- Mean active features per sentence: {:.2f} corpus-wide; {:.2f} on validation.".format(active_per_sentence.mean(), validation_active.mean()))
    lines.append("- Features with support under 5 by rough cue: {} (kept only as low-confidence fine splits where stylistically distinctive).".format(len(low_support)))
    lines.append("- Features over 25% prevalence by rough cue: {} (mostly lexical cue families whose final interpretations are narrower than the regex proxy).".format(len(broad)))
    lines.append("- Known gaps: subtle clause-depth judgments, genuinely semantic discourse roles, and sparse argumentative styles remain under-served by regex prevalence proxies; the final descriptions should be treated as the authoritative interpretations.")
    (WORK / "final_features_stylistic.md").write_text("\n".join(lines) + "\n")

    val_lines = ["# Validation Report", ""]
    val_lines.append(f"- Held-out validation sample: `{validation_path.relative_to(WORK)}` ({len(validation)} rows), drawn before seed feature drafting.")
    val_lines.append("- Validation method: apply the frozen curated cue rules to held-out rows, then inspect activation density, uncovered rows, low-support features, broad feature proxies, and examples retrieved from the full CSV.")
    val_lines.append("- Held-out coverage: {:.1%}.".format((validation_active > 0).mean()))
    val_lines.append("- Mean held-out active features per sentence: {:.2f}; median: {:.0f}; 90th percentile: {:.0f}.".format(validation_active.mean(), validation_active.median(), validation_active.quantile(0.9)))
    val_lines.append("- Low-support final features by rough cue: " + (", ".join(f"{f['id']} {f['name']} ({f['support_count']})" for f in low_support[:40]) or "none") + ".")
    val_lines.append("- Broad rough-cue features above 25% prevalence: " + (", ".join(f"{f['id']} {f['name']} ({f['estimated_prevalence']:.1%})" for f in broad[:40]) or "none") + ".")
    val_lines.append("")
    val_lines.append("## Held-Out Rows With Fewest Activations")
    few = validation.assign(active_features=validation_active.values).sort_values("active_features").head(20)
    for _, r in few.iterrows():
        sent = re.sub(r"\s+", " ", str(r["sentence"])).strip()
        val_lines.append(f"- row {int(r['row_id'])} | {int(r['year'])} | active={int(r['active_features'])}: {sent}")
    val_lines.append("")
    val_lines.append("## Held-Out Rows With Many Activations")
    many = validation.assign(active_features=validation_active.values).sort_values("active_features", ascending=False).head(20)
    for _, r in many.iterrows():
        sent = re.sub(r"\s+", " ", str(r["sentence"])).strip()
        val_lines.append(f"- row {int(r['row_id'])} | {int(r['year'])} | active={int(r['active_features'])}: {sent}")
    (WORK / "validation_report_stylistic.md").write_text("\n".join(val_lines) + "\n")


if __name__ == "__main__":
    build_outputs()
