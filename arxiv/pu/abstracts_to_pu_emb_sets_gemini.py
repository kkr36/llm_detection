import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
import time
import os

def generate_embedding(inputs, client, model_id="gemini-embedding-001",
                       max_retries=5, base_delay=3.0):
    """
    Generate embeddings with exponential backoff on API failure.
    """
    input_texts, input_labels, input_indices = inputs

    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=model_id,
                contents=input_texts
            )
            # Success — break the retry loop
            break
        except Exception as e:
            if attempt == max_retries - 1:
                # Last try — raise the error
                raise
            sleep_time = base_delay * (2 ** attempt)
            print(f"Error: {e}. Retrying in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    embeddings = result.embeddings
    just_embeddings = np.array([embedding.values for embedding in embeddings])
    just_labels = np.array(input_labels).reshape(-1, 1)
    just_indices = np.array(input_indices).reshape(-1,1)

    res = np.hstack([just_embeddings, just_labels, just_indices])
    return res

if __name__ == "__main__":
    # train_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    # train_years = [2021,2022]
    # train_years = [2023]
    subsample_size = 2500
    train_years = list(range(2024,2026))
    for train_year in train_years:
        subsample_path = f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{train_year}_ai_cs._{subsample_size}.parquet"
        # subsample_path = f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{train_year}_ai_cs._5000.parquet" if os.path.exists(f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{train_year}_ai_cs._5000.parquet") else f"/share/garg/arxiv_kaggle/train/arxiv_tokenized_{train_year}_ai_cs._2500.parquet"
        raw_data = pd.read_parquet(subsample_path).dropna(subset=["ai_sentence", "human_sentence"])
        ai_df = pd.DataFrame({
            "text": raw_data["ai_sentence"].apply(lambda x: " ".join(x)),
            "label": 1,
            "index": raw_data["ai_index"]
        })
        human_df = pd.DataFrame({
            "text": raw_data["human_sentence"].apply(lambda x: " ".join(x)),
            "label": -1,
            "index": raw_data["human_index"]
        })
        # assert(max(human_df['index']) == max(ai_df['index']))
        num_abs = max(human_df['index'])+1

        ### TODO: make sure indices aligned properly, sample by index not generally ###

        human_df = human_df[human_df["text"].str.strip() != '']
        # ai_df = ai_df.sample(n=len(human_df), random_state=42).reset_index(drop=True)
        # assert(len(ai_df) == len(human_df))
        combined_df = pd.concat([ai_df, human_df], ignore_index=True)

        texts = combined_df["text"].tolist()
        labels = combined_df["label"].tolist()
        indices = combined_df['index'].tolist()
        # texts = combined_df["text"].tolist()[:100] + combined_df["text"].tolist()[-100:]
        # labels = combined_df["label"].tolist()[:100] + combined_df["label"].tolist()[-100:]
        # indices = combined_df["index"].tolist()[:100] + combined_df["index"].tolist()[-100:]

        # chunk into 100s
        text_chunks = [texts[i:i+100] for i in range(0, len(texts), 100)]
        label_chunks = [labels[i:i+100] for i in range(0, len(labels), 100)]
        index_chunks = [indices[i:i+100] for i in range(0, len(indices), 100)]

        # import pdb; pdb.set_trace()

        embeddings_chunked = []
        max_workers = 5  # tune based on rate limits

        import json

        with open("/home/kkr36/creds.json", "r") as f:
            keys = json.load(f)

        gemini_key = keys["gemini_api_key"]

        client = genai.Client(api_key = gemini_key)

        for t in tqdm(list(zip(text_chunks, label_chunks, index_chunks))):
            chunk = generate_embedding(t, client)
            embeddings_chunked.append(chunk)

        embeddings = np.vstack(embeddings_chunked) # (n,3074); 3072 embeddings, 1 label, 1 index

        # 60 pct of all for train, 20 pct of all for cal, 20 pct of human for eval
        abs_indices = np.arange(num_abs)
        np.random.seed(42)
        abs_indices = np.random.choice(abs_indices,size=num_abs,replace=False) # shuffle
        train_indices = abs_indices[:int(.6*num_abs)]
        cal_indices = abs_indices[int(.6*num_abs):int(.8*num_abs)]
        test_indices = abs_indices[int(.8*num_abs):]

        train_set = embeddings[np.isin(embeddings[:, -1], train_indices)]
        cal_set   = embeddings[np.isin(embeddings[:, -1], cal_indices)]
        test_set  = embeddings[np.isin(embeddings[:, -1], test_indices)]
        train_set, cal_set, test_set = train_set[:,:-1], cal_set[:,:-1], test_set[:,:-1]
    
        test_human = test_set[test_set[:, -1] == -1]
        test_ai    = test_set[test_set[:, -1] ==  1]

        np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_train.npy", train_set)
        np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_cal.npy", cal_set)
        np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_test_0.npy", test_human)

        for alpha in [.05,.1,.2,.3,.5]: # replace alpha pct of the human data with ai sentences (keep overall sample size the same)
            num_human = int(len(test_human) * (1-alpha))
            num_ai = int(len(test_human) * alpha)
            assert(num_human + num_ai - len(test_human) in {0,-1})
            human_set = test_human[:num_human]
            ai_set = test_ai[:num_ai]
            test_alpha = np.vstack([human_set, ai_set])

            # import pdb; pdb.set_trace()

            np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_test_{alpha}.npy", test_alpha)

        # ai_np_df = embeddings[:len(embeddings)//2]
        # human_np_df = embeddings[len(embeddings)//2:]
        # assert(len(ai_np_df) == len(human_np_df))

        # np.random.seed(42)  # for reproducibility

        # # shuffle once so a and b stay aligned
        # n = len(ai_np_df)
        # idx = np.arange(n)
        # np.random.shuffle(idx)

        # # split indices
        # train_end = int(0.6 * n)
        # val_end = int(0.8 * n)

        # train_idx = idx[:train_end]
        # val_idx   = idx[train_end:val_end]
        # test_idx  = idx[val_end:]  # for test, only one list

        # # create splits
        # train = np.vstack([human_np_df[train_idx], ai_np_df[train_idx]])
        # cal = np.vstack([human_np_df[val_idx], ai_np_df[val_idx]])
        # human_test = human_np_df[test_idx]  # only one list for test
        # print(train.shape, cal.shape, human_test.shape)
        # # import pdb; pdb.set_trace()

        # np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_train.npy", train)
        # np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_cal.npy", cal)
        # np.save(f"/share/garg/arxiv_kaggle/pu/{train_year}_test.npy", human_test)

    # import time
    # t1 = time.time()
    # bob = generate_embedding(texts, client)
    # t2 = time.time()
    # diff = t2 - t1
    # import pdb; pdb.set_trace()

    # for i in tqdm(list(range(len(train_df)))):
    #     row = train_df.iloc[i]
    #     text, label = row['text'], row['label']
    #     emb = generate_embedding(text, client)