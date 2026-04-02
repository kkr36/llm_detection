### DO NOT OVERWRITE, EDIT, OR TOUCH ANYTHING IN THIS FILE ###
# must be run using conda env *llm_master*!
# Batch version of inner_loop.py: uses Gemini Batch API to process all abstracts
# at each strategy step simultaneously, then chains steps sequentially.
# Total API round-trips: N_steps batch submissions (vs N_abstracts * N_steps live calls).

import json
import time
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from google.genai import types
from gemini_api import client  # reuse initialized client from gemini_api.py
from util import predict_with_backoff
from strategy import CURRENT_STRATEGY, CURRENT_TIMESTEP
from pangram import Pangram

sys.path.insert(0, "/home/kkr36/llm_detection/arxiv/pu/lipton/PU_learning")

with open("/home/kkr36/creds.json", 'r') as handle:
    pangram_api_key = json.load(handle)['pangram_api_key']
pangram_client = Pangram(api_key=pangram_api_key)

llm_labels = ["Gemini 3 Preview"]
BATCH_MODEL = "gemini-3-pro-preview"

split = "val"
to_rewrite = 15
output_csv = f"results_{CURRENT_TIMESTEP}_batch.csv"


# ---------------------------------------------------------------------------
# StepwiseMock: intercepts strategy execution to capture (context, prompt)
# at a target step, replaying previously-computed results for prior steps.
# ---------------------------------------------------------------------------

class _StopExecution(Exception):
    pass


class StepwiseMock:
    """
    Passed as prompt_model to a strategy function.
    - For steps < target_step: returns the pre-computed result from previous_results.
    - At target_step: captures (context, prompt) and raises _StopExecution.
    """
    def __init__(self, target_step: int, previous_results: list):
        self.step = 0
        self.target_step = target_step
        self.previous_results = previous_results
        self.captured = None

    def __call__(self, context: str, prompt: str) -> str:
        self.step += 1
        if self.step < self.target_step:
            return self.previous_results[self.step - 1]
        elif self.step == self.target_step:
            self.captured = (context, prompt)
            raise _StopExecution()


class CountingMock:
    """Counts how many prompt_model calls a strategy makes."""
    def __init__(self):
        self.count = 0

    def __call__(self, context: str, prompt: str) -> str:
        self.count += 1
        return ""


def count_strategy_steps(strategy_fn, abstract: str, label: str) -> int:
    mock = CountingMock()
    strategy_fn(mock, abstract, label)
    return mock.count


def extract_step_inputs(strategy_fn, abstracts, label, step_num, all_results):
    """
    For each abstract, run strategy up to step_num using StepwiseMock,
    capturing the (context, prompt) pair that would be sent to the API.
    Returns a list of (context, prompt) tuples in abstract order.
    """
    step_inputs = []
    for i, abstract in enumerate(abstracts):
        mock = StepwiseMock(target_step=step_num, previous_results=all_results[i])
        try:
            strategy_fn(mock, abstract, label)
        except _StopExecution:
            step_inputs.append(mock.captured)
    return step_inputs


# ---------------------------------------------------------------------------
# Batch submission: one batch job per strategy step
# ---------------------------------------------------------------------------

def submit_batch_step(step_inputs: list, step_num: int) -> list:
    """
    Submit all (context, prompt) pairs for a single strategy step as one batch job.
    Polls until complete, then returns list of response strings (None on per-request error).
    """
    inline_requests = [
        types.InlinedRequest(
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=context,
                max_output_tokens=2500,
                temperature=0,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        )
        for context, prompt in step_inputs
    ]

    src = types.BatchJobSource(inlined_requests=inline_requests)
    batch_job = client.batches.create(
        model=BATCH_MODEL,
        src=src,
    )
    print(f"  [step {step_num}] Batch job created: {batch_job.name}, polling...")

    while not batch_job.done:
        time.sleep(30)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"  [step {step_num}] State: {batch_job.state}")

    if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
        raise RuntimeError(
            f"Batch job failed: state={batch_job.state}, error={batch_job.error}"
        )

    outputs = []
    for inlined_resp in batch_job.dest.inlined_responses:
        if inlined_resp.error:
            print(f"  Response error: {inlined_resp.error}")
            outputs.append(None)
        else:
            outputs.append(inlined_resp.response.text)
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"starting batch generation for t={CURRENT_TIMESTEP}")

    subsample_size = 20000
    category = "cs."
    train_year = 2010
    val_start = 2500
    test_start = 2600

    start_idx = val_start if split == "val" else test_start
    assert val_start + to_rewrite <= test_start if split == "val" else True

    fpath = (
        f"/share/garg/arxiv_kaggle/multillm/data_raw/"
        f"arxiv_{train_year}_ai_{category}_{subsample_size//2}_fronthalf.parquet"
    )
    arxiv_data = pd.read_parquet(fpath)
    human_abstracts = arxiv_data.iloc[start_idx:start_idx + to_rewrite]["human_abstract"].tolist()
    rng = np.random.default_rng(seed=CURRENT_TIMESTEP)
    rng.shuffle(human_abstracts)
    llms = [i % len(llm_labels) for i in range(len(arxiv_data))]
    label = llm_labels[0]

    # Auto-detect number of steps in the current strategy
    num_steps = count_strategy_steps(CURRENT_STRATEGY, human_abstracts[0], label)
    print(f"Strategy has {num_steps} steps; submitting {num_steps} sequential batch jobs.")

    # all_results[i] accumulates step outputs for abstract i
    all_results = [[] for _ in range(to_rewrite)]

    for step in range(1, num_steps + 1):
        print(f"\nStep {step}/{num_steps}: extracting inputs...")
        step_inputs = extract_step_inputs(
            CURRENT_STRATEGY, human_abstracts, label, step, all_results
        )
        print(f"Step {step}/{num_steps}: submitting batch of {len(step_inputs)} requests...")
        outputs = submit_batch_step(step_inputs, step_num=step)
        for i, out in enumerate(outputs):
            all_results[i].append(out if out is not None else "")

    new_text = [all_results[i][-1] for i in range(to_rewrite)]
    human_text = human_abstracts

    abstract_dict = {
        "human": [],
        "mirroring_llm": [],
        f"mirror_{CURRENT_TIMESTEP}": [],
    }
    for i, (human_abstract, mirror_t) in enumerate(zip(human_text, new_text)):
        abstract_dict["human"].append(human_abstract)
        abstract_dict[f"mirror_{CURRENT_TIMESTEP}"].append(mirror_t)
        abstract_dict["mirroring_llm"].append(llm_labels[llms[i]])

    # Pangram scoring — identical to inner_loop.py
    failed_requests = []
    final_dict = {
        "fraction_ai": [],
        "fraction_ai_assisted": [],
        "fraction_human": [],
        "num_ai_segments": [],
        "window_labels": [],
        "window_ai_assistance_scores": [],
        "window_confidences": [],
    }

    for i, text in tqdm(enumerate(new_text)):
        if len(text) < 5 or "sorry" in text.lower():
            result = {
                "fraction_ai": 50,
                "fraction_ai_assisted": 50,
                "fraction_human": 50,
                "num_ai_segments": 50,
                "window_labels": [50],
                "window_ai_assistance_scores": [50],
                "window_confidences": [50],
            }
        else:
            result = predict_with_backoff(pangram_client, text)

        if result is None or len(result) == 0:
            print(f"failed {i}")
            failed_requests.append(i)
            continue

        try:
            final_dict['fraction_ai'].append(result.get('fraction_ai', np.nan))
            final_dict['fraction_ai_assisted'].append(result.get('fraction_ai_assisted', np.nan))
            final_dict['fraction_human'].append(result.get('fraction_human', np.nan))
            final_dict['num_ai_segments'].append(result.get('num_ai_segments', np.nan))

            window_labels = []
            window_ai_assistance_scores = []
            window_confidences = []
            for window in result.get('windows', []):
                window_labels.append(window.get('label', None))
                window_ai_assistance_scores.append(window.get('ai_assistance_score', np.nan))
                window_confidences.append(window.get('confidence', np.nan))

            final_dict['window_labels'].append(window_labels)
            final_dict['window_ai_assistance_scores'].append(window_ai_assistance_scores)
            final_dict['window_confidences'].append(window_confidences)

        except Exception as e:
            print(f"Failed to process result for {i}: {e}")
            failed_requests.append(i)
            continue

    for key in final_dict:
        abstract_dict[key] = final_dict[key]

    t_csv = pd.DataFrame(abstract_dict)
    t_csv.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
