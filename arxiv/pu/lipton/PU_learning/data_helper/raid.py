import numpy as np
import pandas as pd
import torch

RAID_PATH = '/share/garg/arxiv_kaggle/raid_train.parquet'
EVAL_FRAC = 0.25

# All attack types in RAID (excluding 'none')
RAID_ATTACKS = [
    'whitespace', 'upper_lower', 'synonym', 'perplexity_misspelling',
    'paraphrase', 'number', 'insert_paragraphs', 'homoglyph',
    'article_deletion', 'alternative_spelling', 'zero_width_space',
]

# Label convention (shared across PU, PN, PNU):
#   label=1 -> confirmed/labeled positive (human, attack='none')
#   label=2 -> confirmed/labeled negative (non-human, attack='none')  [PNU only]
#   label=0 -> unlabeled pool / labeled negative for PN


class RAIDBERTData(torch.utils.data.Dataset):
    """Dataset for PU and PN tasks on RAID."""
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        encodings = transform(data)

        p_data_idx = np.where(labels == 1)[0]
        n_data_idx = np.where(labels == 0)[0]

        self.p_data = encodings[p_data_idx, :, :]
        self.n_data = encodings[n_data_idx, :, :]
        self.labels = labels
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)


class RAIDBERTData_PNU(torch.utils.data.Dataset):
    """Dataset for PNU tasks on RAID; supports label=2 for labeled negatives."""
    def __init__(self, data, labels, transform):
        labels = np.array(labels)
        encodings = transform(data)

        p_data_idx  = np.where(labels == 1)[0]
        ln_data_idx = np.where(labels == 2)[0]
        n_data_idx  = np.where(labels == 0)[0]

        self.p_data  = encodings[p_data_idx,  :, :]
        self.ln_data = encodings[ln_data_idx, :, :]
        self.n_data  = encodings[n_data_idx,  :, :]
        self.labels = labels
        self.transform = None
        self.target_transform = None

    def __len__(self):
        return len(self.labels)


def _load_raid_train(seed):
    """Shuffle RAID, hold out EVAL_FRAC (25%) for evaluation, return the train portion."""
    df = pd.read_parquet(RAID_PATH)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * (1 - EVAL_FRAC))
    return df.iloc[:n_train].reset_index(drop=True)


def _human_none_slices(none_human, n_known, n_cal_known, n_u_train, n_u_cal):
    """
    Compute sequential, non-overlapping row slice boundaries for human/attack='none' rows.
    Proportionally scales all four buckets when there are fewer rows than requested.

    Returns (a, b, c, d) such that:
      [0 .. a)   -> train known positives        (n_known)
      [a .. b)   -> cal known positives           (n_cal_known)
      [b .. c)   -> train unlabeled positives     (n_u_train)
      [c .. d)   -> cal unlabeled positives       (n_u_cal)
    """
    available = len(none_human)
    total_requested = n_known + n_cal_known + n_u_train + n_u_cal
    if total_requested == 0:
        return 0, 0, 0, 0
    scale = min(1.0, available / total_requested)
    a = int(n_known     * scale)
    b = a + int(n_cal_known * scale)
    c = b + int(n_u_train   * scale)
    d = available  # give all remaining rows to cal unlabeled positives
    return a, b, c, d


def _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed):
    """
    From u_pos_pool and u_neg_pool, select texts so that
    len(u_pos) / (len(u_pos) + len(u_neg)) == alpha.
    """
    T_pos = len(u_pos_pool) / alpha       if alpha > 0 else float('inf')
    T_neg = len(u_neg_pool) / (1 - alpha) if alpha < 1 else float('inf')
    T = int(min(T_pos, T_neg))
    n_pos = int(alpha * T)
    n_neg = T - n_pos
    rng = np.random.default_rng(seed)
    u_pos = list(rng.choice(u_pos_pool, size=min(n_pos, len(u_pos_pool)), replace=False))
    u_neg = list(rng.choice(u_neg_pool, size=min(n_neg, len(u_neg_pool)), replace=False))
    return u_pos, u_neg


def _get_attacked_llm(train_df, attack, seed, start=0, n=20000):
    """
    Return up to n generation texts from non-human rows with the given attack.
    If attack == 'all', draw an even mix from every attack type.
    """
    if attack == 'all':
        texts = []
        per_atk = n // len(RAID_ATTACKS)
        per_atk_start = start // len(RAID_ATTACKS)
        for atk in RAID_ATTACKS:
            subset = (train_df[(train_df['attack'] == atk) & (train_df['model'] != 'human')]
                      .sample(frac=1, random_state=seed)
                      .reset_index(drop=True))
            texts.extend(subset.iloc[per_atk_start:per_atk_start + per_atk]['generation'].tolist())
        return texts
    else:
        subset = (train_df[(train_df['attack'] == attack) & (train_df['model'] != 'human')]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))
        return subset.iloc[start:start + n]['generation'].tolist()


def read_raid_PU(attack, alpha, split, seed):
    """
    PU learning on the RAID dataset.

    Source domain (labeled positives): human rows where attack == 'none'.
    Test domain (unlabeled):
      - unlabeled positives: human rows where attack == 'none'
      - unlabeled negatives: non-human rows where attack == `attack`
        (if attack == 'all', an even mix of each attack type)

    Training set:
      - 20000 known positives (human, attack='none')
      - 20000 unlabeled negatives (non-human, attack==attack)
      - 20000 unlabeled positives (human, attack='none'), balanced with alpha
    Calibration set:
      - 5000 known positives
      - 10000 unlabeled (same composition, balanced with alpha)

    # max human rows across all attack types in training: int(160452 * 0.75) = 120339
    # max human rows with attack='none' in training: ~10028

    attack : str   specific attack name (e.g. 'whitespace') or 'all'
    alpha  : float fraction of the unlabeled pool that is human (positive)
    split  : str   'train' or 'cal'
    seed   : int

    Returns texts, labels
      label=1 known positive, label=0 unlabeled
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN         = 20000
    N_CAL_KNOWN     = 5000
    N_UNLABELED     = 20000
    N_CAL_UNLABELED = 10000

    none_human = (train_df[(train_df['attack'] == 'none') & (train_df['model'] == 'human')]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    # Sequential slice allocation (avoids train/cal row overlap); proportionally scaled
    # if fewer than N_KNOWN+N_CAL_KNOWN+N_UNLABELED+N_CAL_UNLABELED rows are available:
    #   [0 .. a)  -> train known positives
    #   [a .. b)  -> cal known positives
    #   [b .. c)  -> train unlabeled positives
    #   [c .. d)  -> cal unlabeled positives
    a, b, c, d = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, N_CAL_UNLABELED)

    train_u_neg = _get_attacked_llm(train_df, attack, seed, start=0, n=N_UNLABELED)
    cal_u_neg   = _get_attacked_llm(train_df, attack, seed, start=N_UNLABELED, n=N_CAL_UNLABELED)

    if split == 'train':
        known_pos_texts = none_human.iloc[:a]['generation'].tolist()
        u_pos_pool      = none_human.iloc[b:c]['generation'].tolist()
        u_neg_pool      = train_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID PU train | known_pos={len(known_pos_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = known_pos_texts + u_pos_texts + u_neg_texts
        labels = [1] * len(known_pos_texts) + [0] * (len(u_pos_texts) + len(u_neg_texts))

    else:  # cal
        known_pos_texts = none_human.iloc[a:b]['generation'].tolist()
        u_pos_pool      = none_human.iloc[c:d]['generation'].tolist()
        u_neg_pool      = cal_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID PU cal | known_pos={len(known_pos_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = known_pos_texts + u_pos_texts + u_neg_texts
        labels = [1] * len(known_pos_texts) + [0] * (len(u_pos_texts) + len(u_neg_texts))

    assert len(texts) == len(labels)
    # import pdb; pdb.set_trace()
    return texts, labels


def read_raid_PN(attack, split, seed):
    """
    PN (supervised) learning on the RAID dataset.

    Trains only on rows where attack == 'none'.
    Labeled positives: same human rows as PU known positives + PU unlabeled positives
                       (20000 + 20000 = 40000; capped at available).
    Labeled negatives: 20000 non-human rows where attack == 'none'.

    # max human rows across all attack types in training: int(160452 * 0.75) = 120339

    attack : str   passed through for row-slice alignment with PU (not used to filter negatives)
    split  : str   'train' or 'cal'
    seed   : int

    Returns texts, labels
      label=1 positive (human), label=0 negative (non-human)
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN         = 20000
    N_CAL_KNOWN     = 5000
    N_UNLABELED     = 20000
    N_LABELED_NEG   = 20000
    N_CAL_NEG       = 5000

    none_human = (train_df[(train_df['attack'] == 'none') & (train_df['model'] == 'human')]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    none_llm = (train_df[(train_df['attack'] == 'none') & (train_df['model'] != 'human')]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True))

    # Same slice layout as PU for human rows (proportionally scaled when data is scarce):
    #   [0 .. a)  -> train known positives (also PU train known)
    #   [a .. b)  -> cal positives
    #   [b .. c)  -> train "unlabeled" positives (now labeled positive in PN)
    a, b, c, _ = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, 0)

    if split == 'train':
        # PN labeled positives = PU train known + PU train unlabeled positives
        pos_texts = none_human.iloc[:a]['generation'].tolist()
                    #  + none_human.iloc[b:c]['generation'].tolist())
        neg_texts = none_llm.iloc[:N_LABELED_NEG]['generation'].tolist()

        print(f"RAID PN train | pos={len(pos_texts)} | neg={len(neg_texts)}")

        texts  = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    else:  # cal
        pos_texts = none_human.iloc[a:b]['generation'].tolist()
        neg_texts = none_llm.iloc[N_LABELED_NEG:N_LABELED_NEG + N_CAL_NEG]['generation'].tolist()

        print(f"RAID PN cal | pos={len(pos_texts)} | neg={len(neg_texts)}")

        texts  = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    assert len(texts) == len(labels)
    return texts, labels


def _get_shifted_llm(train_df, shift_col, shift_val, seed, start=0, n=20000):
    """
    Return up to n generation texts from non-human rows where attack=='none'
    and shift_col==shift_val.
    """
    subset = (train_df[
        (train_df['attack'] == 'none') &
        (train_df['model'] != 'human') &
        (train_df[shift_col] == shift_val)
    ]
    .sample(frac=1, random_state=seed)
    .reset_index(drop=True))
    return subset.iloc[start:start + n]['generation'].tolist()


def read_raid_shift_PU(shift_col, source_val, target_val, alpha, split, seed):
    """
    PU learning with distribution shift on RAID dataset.

    All rows are pre-filtered to attack=='none'.
    Labeled (known) positives: human rows (attack='none'); when shift_col=='domain',
      further filtered to domain==source_val.
    Unlabeled positives: same human pool as known positives.
    Unlabeled negatives: non-human rows (attack='none', shift_col==target_val).

    shift_col  : str   column to shift on ('repetition_penalty', 'decoding', 'domain', 'model')
    source_val : str   value for the source/labeled distribution (used to filter known positives
                       when shift_col=='domain')
    target_val : str   value for the target/test-time LLM distribution
    alpha      : float fraction of unlabeled pool that is human (positive)
    split      : str   'train' or 'cal'
    seed       : int
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN         = 20000
    N_CAL_KNOWN     = 5000
    N_UNLABELED     = 20000
    N_CAL_UNLABELED = 10000

    human_mask = (train_df['attack'] == 'none') & (train_df['model'] == 'human')
    if shift_col == 'domain':
        human_mask = human_mask & (train_df['domain'] == source_val)
    none_human = (train_df[human_mask]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    a, b, c, d = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, N_CAL_UNLABELED)

    train_u_neg = _get_shifted_llm(train_df, shift_col, target_val, seed, start=0, n=N_UNLABELED)
    cal_u_neg   = _get_shifted_llm(train_df, shift_col, target_val, seed, start=N_UNLABELED, n=N_CAL_UNLABELED)

    if split == 'train':
        known_pos_texts = none_human.iloc[:a]['generation'].tolist()
        u_pos_pool      = none_human.iloc[b:c]['generation'].tolist()
        u_neg_pool      = train_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID shift PU train ({shift_col}: {source_val}->{target_val}) | known_pos={len(known_pos_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = known_pos_texts + u_pos_texts + u_neg_texts
        labels = [1] * len(known_pos_texts) + [0] * (len(u_pos_texts) + len(u_neg_texts))

    else:  # cal
        known_pos_texts = none_human.iloc[a:b]['generation'].tolist()
        u_pos_pool      = none_human.iloc[c:d]['generation'].tolist()
        u_neg_pool      = cal_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID shift PU cal ({shift_col}: {source_val}->{target_val}) | known_pos={len(known_pos_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = known_pos_texts + u_pos_texts + u_neg_texts
        labels = [1] * len(known_pos_texts) + [0] * (len(u_pos_texts) + len(u_neg_texts))

    # if shift_col == "model": import pdb; pdb.set_trace()

    assert len(texts) == len(labels)
    return texts, labels


def read_raid_shift_PN(shift_col, source_val, split, seed):
    """
    PN (supervised) learning with distribution shift on RAID dataset.

    All rows are pre-filtered to attack=='none'.
    Labeled positives: human rows (attack='none').
    Labeled negatives: non-human rows (attack='none', shift_col==source_val).

    shift_col  : str   column to shift on
    source_val : str   value for the source/labeled LLM distribution
    split      : str   'train' or 'cal'
    seed       : int
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN       = 20000
    N_CAL_KNOWN   = 5000
    N_UNLABELED   = 20000
    N_LABELED_NEG = 20000
    N_CAL_NEG     = 5000

    # none_human = (train_df[(train_df['attack'] == 'none') & (train_df['model'] == 'human')]
    #               .sample(frac=1, random_state=seed)
    #               .reset_index(drop=True))
    human_mask = (train_df['attack'] == 'none') & (train_df['model'] == 'human')
    if shift_col == 'domain':
        human_mask = human_mask & (train_df['domain'] == source_val)
    none_human = (train_df[human_mask]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    source_llm = (train_df[
        (train_df['attack'] == 'none') &
        (train_df['model'] != 'human') &
        (train_df[shift_col] == source_val)
    ]
    .sample(frac=1, random_state=seed)
    .reset_index(drop=True))

    a, b, c, _ = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, 0)

    if split == 'train':
        pos_texts = none_human.iloc[:a]['generation'].tolist()
        neg_texts = source_llm.iloc[:N_LABELED_NEG]['generation'].tolist()

        print(f"RAID shift PN train ({shift_col}={source_val}) | pos={len(pos_texts)} | neg={len(neg_texts)}")

        texts  = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    else:  # cal
        pos_texts = none_human.iloc[a:b]['generation'].tolist()
        neg_texts = source_llm.iloc[N_LABELED_NEG:N_LABELED_NEG + N_CAL_NEG]['generation'].tolist()

        print(f"RAID shift PN cal ({shift_col}={source_val}) | pos={len(pos_texts)} | neg={len(neg_texts)}")

        texts  = pos_texts + neg_texts
        labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    assert len(texts) == len(labels)
    return texts, labels


def read_raid_shift_PNU(shift_col, source_val, target_val, alpha, split, seed):
    """
    PNU learning with distribution shift on RAID dataset.

    All rows are pre-filtered to attack=='none'.
    Labeled positives (label=1): human rows (attack='none').
    Labeled negatives (label=2): non-human rows (attack='none', shift_col==source_val).
    Unlabeled positives (label=0): human rows (attack='none').
    Unlabeled negatives (label=0): non-human rows (attack='none', shift_col==target_val).

    shift_col  : str   column to shift on
    source_val : str   value for the source/labeled LLM distribution
    target_val : str   value for the target/unlabeled LLM distribution
    alpha      : float fraction of unlabeled pool that is human (positive)
    split      : str   'train' or 'cal'
    seed       : int
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN         = 20000
    N_CAL_KNOWN     = 5000
    N_UNLABELED     = 20000
    N_CAL_UNLABELED = 10000
    N_LABELED_NEG   = 20000
    N_CAL_NEG       = 5000

    none_human = (train_df[(train_df['attack'] == 'none') & (train_df['model'] == 'human')]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    source_llm = (train_df[
        (train_df['attack'] == 'none') &
        (train_df['model'] != 'human') &
        (train_df[shift_col] == source_val)
    ]
    .sample(frac=1, random_state=seed)
    .reset_index(drop=True))

    a, b, c, d = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, N_CAL_UNLABELED)

    train_u_neg = _get_shifted_llm(train_df, shift_col, target_val, seed, start=0, n=N_UNLABELED)
    cal_u_neg   = _get_shifted_llm(train_df, shift_col, target_val, seed, start=N_UNLABELED, n=N_CAL_UNLABELED)

    if split == 'train':
        lp_texts   = none_human.iloc[:a]['generation'].tolist()
        ln_texts   = source_llm.iloc[:N_LABELED_NEG]['generation'].tolist()
        u_pos_pool = none_human.iloc[b:c]['generation'].tolist()
        u_neg_pool = train_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID shift PNU train ({shift_col}: {source_val}->{target_val}) | "
              f"labeled_pos={len(lp_texts)} labeled_neg={len(ln_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = lp_texts + ln_texts + u_pos_texts + u_neg_texts
        labels = ([1] * len(lp_texts)
                  + [2] * len(ln_texts)
                  + [0] * (len(u_pos_texts) + len(u_neg_texts)))

    else:  # cal
        lp_texts   = none_human.iloc[a:b]['generation'].tolist()
        ln_texts   = source_llm.iloc[N_LABELED_NEG:N_LABELED_NEG + N_CAL_NEG]['generation'].tolist()
        u_pos_pool = none_human.iloc[c:d]['generation'].tolist()
        u_neg_pool = cal_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID shift PNU cal ({shift_col}: {source_val}->{target_val}) | "
              f"labeled_pos={len(lp_texts)} labeled_neg={len(ln_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = lp_texts + ln_texts + u_pos_texts + u_neg_texts
        labels = ([1] * len(lp_texts)
                  + [2] * len(ln_texts)
                  + [0] * (len(u_pos_texts) + len(u_neg_texts)))

    assert len(texts) == len(labels)
    return texts, labels


def read_raid_PNU(attack, alpha, split, seed):
    """
    PNU learning on the RAID dataset.

    Labeled positives (label=1): human rows, attack='none'  -- same pool as PN, minus cal holdout.
    Labeled negatives (label=2): non-human rows, attack='none'  -- same as PN.
    Unlabeled (label=0):
      - unlabeled positives: human rows, attack='none' (held out from labeled pool for cal)
      - unlabeled negatives: non-human rows, attack==`attack`
        (if attack == 'all', even mix of all attack types)

    Training:
      - labeled positives: 20000 human/none rows (= PU train known pos)
      - labeled negatives: 20000 non-human/none rows (same as PN)
      - unlabeled positives: 20000 human/none rows (= PU train unlabeled pos)
      - unlabeled negatives: 20000 attacked LLM rows (balanced with alpha)
    Calibration:
      - labeled positives: 5000 human/none rows (= PU cal known pos)

    # max human rows across all attack types in training: int(160452 * 0.75) = 120339

    attack : str   specific attack name or 'all'
    alpha  : float fraction of the unlabeled pool that is human (positive)
    split  : str   'train' or 'cal'
    seed   : int

    Returns texts, labels
      label=1 labeled positive, label=2 labeled negative, label=0 unlabeled
    """
    assert split in ['train', 'cal']
    train_df = _load_raid_train(seed)

    N_KNOWN         = 20000
    N_CAL_KNOWN     = 5000
    N_UNLABELED     = 20000
    N_CAL_UNLABELED = 10000
    N_LABELED_NEG   = 20000
    N_CAL_NEG       = 5000

    none_human = (train_df[(train_df['attack'] == 'none') & (train_df['model'] == 'human')]
                  .sample(frac=1, random_state=seed)
                  .reset_index(drop=True))

    none_llm = (train_df[(train_df['attack'] == 'none') & (train_df['model'] != 'human')]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True))

    # Same human slice layout as PU/PN (proportionally scaled when data is scarce):
    #   [0 .. a)  -> train labeled positives
    #   [a .. b)  -> cal labeled positives (held out from PN labeled pool)
    #   [b .. c)  -> train unlabeled positives
    #   [c .. d)  -> cal unlabeled positives
    a, b, c, d = _human_none_slices(none_human, N_KNOWN, N_CAL_KNOWN, N_UNLABELED, N_CAL_UNLABELED)

    train_u_neg = _get_attacked_llm(train_df, attack, seed, start=0, n=N_UNLABELED)
    cal_u_neg   = _get_attacked_llm(train_df, attack, seed, start=N_UNLABELED, n=N_CAL_UNLABELED)

    if split == 'train':
        lp_texts = none_human.iloc[:a]['generation'].tolist()
        ln_texts = none_llm.iloc[:N_LABELED_NEG]['generation'].tolist()
        u_pos_pool = none_human.iloc[b:c]['generation'].tolist()
        u_neg_pool = train_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID PNU train | labeled_pos={len(lp_texts)} labeled_neg={len(ln_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = lp_texts + ln_texts + u_pos_texts + u_neg_texts
        labels = ([1] * len(lp_texts)
                  + [2] * len(ln_texts)
                  + [0] * (len(u_pos_texts) + len(u_neg_texts)))

    else:  # cal
        lp_texts = none_human.iloc[a:b]['generation'].tolist()
        ln_texts = none_llm.iloc[N_LABELED_NEG:N_LABELED_NEG + N_CAL_NEG]['generation'].tolist()
        u_pos_pool = none_human.iloc[c:d]['generation'].tolist()
        u_neg_pool = cal_u_neg

        u_pos_texts, u_neg_texts = _balance_unlabeled(u_pos_pool, u_neg_pool, alpha, seed)

        print(f"RAID PNU cal | labeled_pos={len(lp_texts)} labeled_neg={len(ln_texts)} | "
              f"alpha={len(u_pos_texts)}/{len(u_pos_texts)+len(u_neg_texts)}")

        texts  = lp_texts + ln_texts + u_pos_texts + u_neg_texts
        labels = ([1] * len(lp_texts)
                  + [2] * len(ln_texts)
                  + [0] * (len(u_pos_texts) + len(u_neg_texts)))

    assert len(texts) == len(labels)
    return texts, labels
