"""Immutable task mapping and prompts for coding-agent annotation."""

from __future__ import annotations

from pathlib import Path


AGENT_JUDGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = Path(
    "/share/garg/arxiv_kaggle/multillm/data_raw/"
    "arxiv_2020_xyz_v2_cs._10000_fronthalf.parquet"
)
DEFAULT_SAMPLE_N = 100
DEFAULT_SAMPLE_SEED = 42
DEFAULT_BATCH_SIZE = 20

# Each rewritten text is judged against the original/reference column named
# alongside it. In particular, later Z rewrites are not compared with the
# human abstract.
COMPARISONS = (
    {"rewrite_col": "rewrite_X", "original_col": "human_abstract"},
    {"rewrite_col": "rewrite_Z_332", "original_col": "rewrite_X"},
    {"rewrite_col": "rewrite_Z_1_PU", "original_col": "rewrite_X"},
    {"rewrite_col": "rewrite_Z_1_PN", "original_col": "rewrite_X"},
    {"rewrite_col": "rewrite_Z_2_PU", "original_col": "rewrite_X"},
    {"rewrite_col": "rewrite_Z_2_PN", "original_col": "rewrite_X"},
)


# These strings are copied verbatim from ../llm_judge.py. Do not edit one
# copy without checking the other: keeping the scoring/prompting system fixed
# is an experimental requirement.
HALLUCINATION_CONTEXT = (
    "You are an expert judge evaluating whether a rewritten text introduces "
    "information that was NOT present in the original text. "
    "Focus only on fabricated or added claims — ignore stylistic differences. "
    "Do not reason at length. Output ONLY a JSON object."
)

HALLUCINATION_PROMPT_TEMPLATE = """\
Original text:
\"\"\"
{original}
\"\"\"

Rewritten text:
\"\"\"
{rewrite}
\"\"\"

Rate how free the rewritten text is from hallucinated or added information not present in the original.
A score of 1.0 means no hallucination at all. A score of 0.0 means the rewrite is entirely composed of fabricated information.
Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}
"""

OMISSION_CONTEXT = (
    "You are an expert judge evaluating whether a rewritten text omits crucial "
    "information that was present in the original text. "
    "Focus only on dropped claims or missing key content — ignore stylistic differences. "
    "Do not reason at length. Output ONLY a JSON object."
)

OMISSION_PROMPT_TEMPLATE = """\
Original text:
\"\"\"
{original}
\"\"\"

Rewritten text:
\"\"\"
{rewrite}
\"\"\"

Rate how completely the rewritten text preserves the crucial information from the original.
A score of 1.0 means nothing important was dropped. A score of 0.0 means the rewrite omits all key information.
Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}
"""

DIRECT_JUDGMENT_METHOD = "direct_semantic_judgment"

ANNOTATION_FIELDS = (
    "task_id",
    "hallucination_score",
    "hallucination_reason",
    "omission_score",
    "omission_reason",
    "annotator",
    "annotation_method",
)
