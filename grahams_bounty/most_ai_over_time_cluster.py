from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler
from Levenshtein import distance as edit_distance

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)
plt.rcParams["font.size"] = 20

# ---- NEW: cluster words by edit distance < n ---- #
def cluster_words(words, max_dist=3):
    clusters = []
    for w in words:
        placed = False
        for cluster in clusters:
            # if close to ANY word in cluster, group them
            if any(edit_distance(w, c) <= max_dist for c in cluster):
                cluster.append(w)
                placed = True
                break
        if not placed:
            clusters.append([w])
    return clusters

if __name__ == "__main__":
    val_years = list(range(2020,2027,1))

    # original list
    most_ai = ['delve', 'intricacies', 'unlock', 'nuanced', 'revolutionize', 'garnered',
               'nuances', 'sparked', 'multifaceted', 'unlocking', 'groundbreaking',
               'showcasing', 'realm', 'elucidating', 'intricate', 'elaborating',
               'appreciation', 'poised', 'revolutionized', 'fascinating',
               'shedding', 'plagued', 'deeper', 'harnessing', 'milestone',
               'meticulous', 'avenues', 'pressing', 'warrants', 'unequivocally',
               'policymakers', 'breakthrough', 'paving', 'govern', 'ultimately',
               'strides', 'highlighting', 'oversight', 'hurdle', 'revolves',
               'cornerstone', 'daunting', 'valuable', 'profound', 'uncover',
               'advancement', 'implications', 'lies', 'reaching', 'prioritize']

    # ---- NEW CLUSTERING ---- #
    max_edit_dist = 3
    clusters = cluster_words(most_ai, max_dist=max_edit_dist)[:20]
    cluster_labels = [" / ".join(c) for c in clusters]   # for legend
    # import pdb; pdb.set_trace()

    counts = {label: [] for label in cluster_labels}

    fig, ax = plt.subplots(figsize=(30, 15))

    for year in tqdm(list(val_years)):
        val_path = f"/share/garg/openreview_data/graham/iclr-dataset/data/tokenized_iclr_{year}_full_abs.parquet"
        val_data = pd.read_parquet(val_path)
        if 'human_sentence' not in val_data.columns:
            val_data['human_sentence'] = val_data['human_abstract']

        exploded = val_data.reset_index().explode("human_sentence")
        unique_per_row = exploded.drop_duplicates(subset=["index", "human_sentence"])
        row_counts = unique_per_row.groupby("human_sentence")["index"].count()
        N = len(val_data)
        percent = (row_counts / N) * 100

        # ---- NEW: cluster-level frequency ---- #
        for cluster, label in zip(clusters, cluster_labels):
            # freq = sum of any word in the cluster
            freq = sum(percent.get(w, 0) for w in cluster)
            counts[label].append(freq)

    # plotting
    for label in cluster_labels:
        plt.plot(val_years, counts[label], marker='o', label=min(label.split(" / "), key=len))

    plt.ylabel('Frequency (% of papers using word)')
    plt.xlabel('Year')
    plt.title(f"Frequency of LLM-y tokens over time @ ICLR")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.savefig("words_over_time_abs_clustered.pdf", format='pdf', bbox_inches="tight")
