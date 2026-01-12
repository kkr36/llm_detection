import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import Counter

# Initialize CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def create_token_frequency_df(df, include_bigrams=True, min_occurrences=2):
    """
    Create a DataFrame with token frequencies and ratios for each label.
    
    Parameters:
    - df: DataFrame with 'code' and 'label' columns
    - include_bigrams: Whether to analyze bigrams in addition to unigrams
    - min_occurrences: Minimum total occurrences across both labels to include
    
    Returns:
    - unigram_df: DataFrame with unigram statistics
    - bigram_df: DataFrame with bigram statistics (if include_bigrams=True)
    """
    
    # Separate by label
    code_label_0 = df[df['label'] == 0]['code'].tolist()
    code_label_1 = df[df['label'] == 1]['code'].tolist()
    
    print(f"Analyzing {len(code_label_0)} samples with label 0")
    print(f"Analyzing {len(code_label_1)} samples with label 1")
    
    # --- UNIGRAM ANALYSIS ---
    tokens_0 = []
    tokens_1 = []
    
    for code in code_label_0:
        tokens = tokenizer.tokenize(code)
        tokens_0.extend(tokens)
    
    for code in code_label_1:
        tokens = tokenizer.tokenize(code)
        tokens_1.extend(tokens)
    
    # Count tokens
    counter_0 = Counter(tokens_0)
    counter_1 = Counter(tokens_1)
    
    # Calculate totals
    total_0 = sum(counter_0.values())
    total_1 = sum(counter_1.values())
    
    # Get all unique tokens
    all_tokens = set(counter_0.keys()) | set(counter_1.keys())
    
    # Build data for DataFrame
    token_data = []
    for token in all_tokens:
        count_0 = counter_0.get(token, 0)
        count_1 = counter_1.get(token, 0)
        total_count = count_0 + count_1
        
        # Skip if below minimum occurrences
        if total_count < min_occurrences:
            continue
        
        # Calculate frequency densities (proportion of all tokens)
        freq_density_0 = count_0 / total_0 if total_0 > 0 else 0
        freq_density_1 = count_1 / total_1 if total_1 > 0 else 0
        
        # Calculate ratio (label_0 / label_1) with smoothing
        ratio_0_to_1 = (freq_density_0 + 1e-10) / (freq_density_1 + 1e-10)
        
        # Log ratio for better interpretability (negative = more in label 1, positive = more in label 0)
        log_ratio = np.log2(ratio_0_to_1)
        
        token_data.append({
            'token': token,
            'count_label_0': count_0,
            'count_label_1': count_1,
            'total_count': total_count,
            'freq_density_label_0': freq_density_0,
            'freq_density_label_1': freq_density_1,
            'ratio_0_to_1': ratio_0_to_1,
            'log2_ratio_0_to_1': log_ratio
        })
    
    # Create DataFrame
    unigram_df = pd.DataFrame(token_data)
    
    # Sort by log ratio (most associated with label 0 first)
    unigram_df = unigram_df.sort_values('log2_ratio_0_to_1', ascending=False).reset_index(drop=True)
    
    print(f"\nCreated unigram DataFrame with {len(unigram_df)} tokens")
    
    if not include_bigrams:
        return unigram_df
    
    # --- BIGRAM ANALYSIS ---
    def get_bigrams(tokens):
        return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    
    bigrams_0 = []
    bigrams_1 = []
    
    for code in code_label_0:
        tokens = tokenizer.tokenize(code)
        if len(tokens) > 1:
            bigrams_0.extend(get_bigrams(tokens))
    
    for code in code_label_1:
        tokens = tokenizer.tokenize(code)
        if len(tokens) > 1:
            bigrams_1.extend(get_bigrams(tokens))
    
    # Count bigrams
    bigram_counter_0 = Counter(bigrams_0)
    bigram_counter_1 = Counter(bigrams_1)
    
    total_bigrams_0 = sum(bigram_counter_0.values())
    total_bigrams_1 = sum(bigram_counter_1.values())
    
    # Get all unique bigrams
    all_bigrams = set(bigram_counter_0.keys()) | set(bigram_counter_1.keys())
    
    # Build data for DataFrame
    bigram_data = []
    for bigram in all_bigrams:
        count_0 = bigram_counter_0.get(bigram, 0)
        count_1 = bigram_counter_1.get(bigram, 0)
        total_count = count_0 + count_1
        
        # Skip if below minimum occurrences
        if total_count < min_occurrences:
            continue
        
        # Calculate frequency densities
        freq_density_0 = count_0 / total_bigrams_0 if total_bigrams_0 > 0 else 0
        freq_density_1 = count_1 / total_bigrams_1 if total_bigrams_1 > 0 else 0
        
        # Calculate ratio
        ratio_0_to_1 = (freq_density_0 + 1e-10) / (freq_density_1 + 1e-10)
        log_ratio = np.log2(ratio_0_to_1)
        
        bigram_data.append({
            'bigram': bigram,
            'count_label_0': count_0,
            'count_label_1': count_1,
            'total_count': total_count,
            'freq_density_label_0': freq_density_0,
            'freq_density_label_1': freq_density_1,
            'ratio_0_to_1': ratio_0_to_1,
            'log2_ratio_0_to_1': log_ratio
        })
    
    # Create DataFrame
    bigram_df = pd.DataFrame(bigram_data)
    bigram_df = bigram_df.sort_values('log2_ratio_0_to_1', ascending=False).reset_index(drop=True)
    
    print(f"Created bigram DataFrame with {len(bigram_df)} bigrams")
    
    return unigram_df, bigram_df


# Example usage:
# unigram_df, bigram_df = create_token_frequency_df(df, include_bigrams=True, min_occurrences=2)
# 
# # View tokens most associated with label 0
# print(unigram_df.head(20))
# 
# # View tokens most associated with label 1
# print(unigram_df.tail(20))
# 
# # Save to CSV
# unigram_df.to_csv('token_frequencies.csv', index=False)
# bigram_df.to_csv('bigram_frequencies.csv', index=False)

# print("Load your DataFrame and call:")
# print("unigram_df, bigram_df = create_token_frequency_df(df)")

test_sample = pd.read_parquet("/home/ubuntu/data/Task_A/test_sample.parquet")
# test = pd.read_parquet("/home/ubuntu/data/Task_A/test.parquet")

# combined = pd.concat([test_sample, test]).reset_index(drop=True)
results = create_token_frequency_df(test_sample, min_occurrences=1)