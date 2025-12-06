most_ai = ['delve', 'intricacies', 'unlock', 'nuanced', 'revolutionize', 'garnered', 'nuances', 'sparked', 'multifaceted', 'unlocking', 'groundbreaking', 'showcasing', 'realm', 'elucidating', 'intricate', 'elaborating', 'appreciation', 'poised', 'revolutionized', 'fascinating', 'shedding', 'plagued', 'deeper', 'harnessing', 'milestone', 'meticulous', 'avenues', 'pressing', 'warrants', 'unequivocally', 'policymakers', 'breakthrough', 'paving', 'govern', 'ultimately', 'strides', 'highlighting', 'oversight', 'hurdle', 'revolves', 'cornerstone', 'daunting', 'valuable', 'profound', 'uncover', 'advancement', 'implications', 'lies', 'reaching', 'prioritize'][:20]

from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from cycler import cycler

plt.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.tab20.colors)

if __name__ == "__main__":
    val_years = list(range(2020,2027,1))

    counts = {word: [] for word in most_ai}
    fig, ax = plt.subplots(figsize=(20, 10))

    for year in tqdm(list(val_years)):
        val_path = f"/share/garg/openreview_data/graham/iclr-dataset/data/tokenized_iclr_{year}_full_abs.parquet"
        # val_path = f"/share/garg/openreview_data/graham/iclr-dataset/data/tokenized_iclr_{year}.parquet"

        val_data = pd.read_parquet(val_path)
        if 'human_sentence' not in val_data.columns:
            val_data['human_sentence'] = val_data['human_abstract']

        exploded = val_data.reset_index().explode("human_sentence")
        unique_per_row = exploded.drop_duplicates(subset=["index", "human_sentence"])
        row_counts = (
            unique_per_row.groupby("human_sentence")["index"]
            .count()
        )
        N = len(val_data)
        percent = row_counts / N
        freqs = {w: percent.get(w, 0)*100 for w in most_ai}
        import pdb; pdb.set_trace()

        for word in most_ai:
            counts[word].append(freqs[word])
    
    for word in most_ai:
        plt.plot(val_years, counts[word], marker='o', label=word)
    
    plt.ylabel('Frequency (% of papers using the word)')
    plt.xlabel('Year')
    plt.title("Frequency of LLM-y words over time @ ICLR")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.savefig("words_over_time_abs.pdf", format='pdf')
            
