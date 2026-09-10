### MOCK VERSION of inner_loop.py
### All LLM API calls are replaced with a mock that returns a default abstract.
# must be run using conda env *llm_master*!

import pandas as pd
from tqdm import tqdm

from strategy import CURRENT_TIMESTEP

MOCK_ABSTRACT = (
    "This paper presents a simple baseline for the problem at hand. "
    "The method is evaluated on standard benchmarks. "
    "Results are mixed, with clear gains in some settings and weaker behavior in others. "
    "The study ends by noting limitations and open questions."
)

llm_labels = ["MOCK"]
output_csv = f"results_{CURRENT_TIMESTEP}_mock.csv"

if __name__ == "__main__":
    print(f"starting generation for t={CURRENT_TIMESTEP} (MOCK MODE)")

    to_rewrite = 10
    arxiv_data = pd.read_csv("results_0_oss_val_50.csv")
    original_text = arxiv_data["mirror_0"].tolist()[:to_rewrite]

    abstract_dict = {
        "original": [],
        "mirroring_llm": [],
        f"mirror_{CURRENT_TIMESTEP}": [],
    }

    for orig_abs in tqdm(original_text):
        abstract_dict["original"].append(orig_abs)
        abstract_dict["mirroring_llm"].append(llm_labels[0])
        abstract_dict[f"mirror_{CURRENT_TIMESTEP}"].append(MOCK_ABSTRACT)

    pd.DataFrame(abstract_dict).to_csv(output_csv, index=False)
    print(f"saved {output_csv}")
