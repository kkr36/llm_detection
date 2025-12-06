import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_pca_human_ai(df, filename, human_col="human_abstract", ai_col="ai_abstract"):
    # ---- 1. Reshape to long format ----
    long_df = pd.DataFrame({
        "text": pd.concat([df[human_col], df[ai_col]], ignore_index=True),
        "author": (["human"] * len(df)) + (["ai"] * len(df))
    })

    # Drop missing
    long_df = long_df.dropna(subset=["text"])

    # ---- 2. Vectorize ----
    tfidf = TfidfVectorizer(min_df=2, stop_words="english")
    X = tfidf.fit_transform(long_df["text"])

    # ---- 3. PCA ----
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X.toarray())

    # ---- 4. Plot ----
    plt.figure(figsize=(8,6))
    for author, color in [("human", "blue"), ("ai", "red")]:
        idx = long_df["author"] == author
        plt.scatter(X2[idx, 0], X2[idx, 1], label=author, alpha=0.7)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Human vs AI Abstracts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf')

# Usage:
# plot_pca_human_ai(your_dataframe)

year = 2020
category = "cs."
subsample_size = 20000
arxiv_path = f"/share/garg/arxiv_kaggle/multillm/data_raw/arxiv_{year}_ai_{category}_{subsample_size}.parquet"
arxiv_data = pd.read_parquet(arxiv_path)

llm_writing = []
llm_cols = ["Llama 3.3 70b Instruct", "Gemini 3 Preview", "GPT OSS 120b", "Gemini 2.5 Flash"]
for i in tqdm(list(i for i in range(len(arxiv_data)))):
    assert(len(arxiv_data.iloc[i][llm_cols[i % 4]]) > 0)
    llm_writing.append(arxiv_data.iloc[i][llm_cols[i % 4]])
arxiv_data['ai_abstract'] = llm_writing

plot_pca_human_ai(arxiv_data, f"{year}_pca.pdf")
