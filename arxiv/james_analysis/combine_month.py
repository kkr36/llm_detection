import json
from collections import defaultdict

by_month_path = "/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_cs._2000_month.json"

# combine by year
with open(by_month_path, 'r') as f:
    data = json.load(f)

new_data = defaultdict(list)
for month in data:
    year = month.split("-")[0]
    new_data[year] += data[month]

with open("/share/garg/arxiv_kaggle/subsamples/arxiv-metadata-oai-snapshot_cs._big_month.json", 'w') as f:
    json.dump(new_data, f)