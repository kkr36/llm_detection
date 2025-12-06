import os
import json
from tqdm import tqdm
from collections import defaultdict
import pdb
import numpy as np

def load_arxiv_snapshot(path, category):
    arxiv_abs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if not line.strip(): continue
            entry = json.loads(line)
            # import pdb; pdb.set_trace()
            if category and category in entry['categories']:
                abstract = entry.get("abstract")
                year = entry.get("update_date").split("-")[0]
                if year and abstract:
                    arxiv_abs[year].append(abstract)
    return arxiv_abs

if __name__ == "__main__":
    category = 'cs.'
    arxiv_path = "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"
    arxiv_data = load_arxiv_snapshot(arxiv_path, category)
    # import pdb; pdb.set_trace()
    subsample_size = 20000
    new_arxiv_data = {}
    for year in tqdm(arxiv_data):
        np.random.seed(42)
        year_data = arxiv_data[year]
        subsample = np.random.choice(year_data,min(len(year_data),subsample_size),replace=False)
        new_arxiv_data[year] = subsample.tolist()
    import pdb; pdb.set_trace()
    with open(f"/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_{category}_{subsample_size}.json", "w") as f:
        json.dump(new_arxiv_data, f)