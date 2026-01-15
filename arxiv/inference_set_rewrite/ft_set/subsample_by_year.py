import os
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import re

def load_arxiv_snapshot(path, category):
    arxiv_abs = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading arxiv data"):
            if not line.strip(): continue
            entry = json.loads(line)
            if category and category in entry['categories']:
                abstract = entry.get("abstract")
                year = entry.get("update_date").split("-")[0]
                if year and abstract:
                    arxiv_abs[year].append(abstract)
    return arxiv_abs

def split_sentences(text):
    """
    More robust sentence splitting that handles:
    - Periods, question marks, exclamation marks
    - Abbreviations (e.g., Dr., et al., i.e.)
    - Decimals in numbers
    """
    # Simple regex-based sentence splitter
    # Splits on . ! ? followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]

def create_expansion_example(abstract):
    """Create instruction-tuning example from abstract."""
    sentences = split_sentences(abstract)
    
    # Skip if too few sentences or very short
    if len(sentences) < 2 or len(abstract.split()) < 50:
        return None
    
    # Always use beginning of abstract (continuation task)
    # max_sentences = min(3, len(sentences) - 1)
    # num_sentences = np.random.randint(1, max_sentences + 1)

    # 50% use 1 sentence, 30% use 2, 20% use 3
    max_sentences = min(3, len(sentences) - 1)
    # Weighted selection: prefer fewer sentences (harder task)
    probs = [0.45, 0.3, 0.25]
    probs = probs[:max_sentences]
    probs = np.array(probs) / np.sum(probs)  # Normalize to sum to 1
    
    num_sentences = np.random.choice(range(1, max_sentences + 1), p=probs)

    prompt_text = ' '.join(sentences[:num_sentences])
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are writing arxiv abstracts in the style of a researcher. Use appropriate academic language and structure."
            },
            {
                "role": "user",
                "content": f"Complete this abstract:\n\n{prompt_text}"
            },
            {
                "role": "assistant",
                "content": abstract
            }
        ]
    }

if __name__ == "__main__":
    category = 'cs.'
    arxiv_path = "/share/garg/arxiv_kaggle/arxiv-metadata-oai-snapshot.json"
    
    # Years to sample from
    years = [str(year) for year in range(2014, 2021)]  # 2014-2020
    samples_per_year = 7000
    
    print("Loading arxiv data...")
    arxiv_data = load_arxiv_snapshot(arxiv_path, category)
    
    # Get the original sampled indices to exclude them
    print("Identifying complementary samples...")
    complementary_data = {}
    
    for year in tqdm(years, desc="Processing years"):
        if year not in arxiv_data:
            print(f"Warning: No data for year {year}")
            continue
            
        year_data = arxiv_data[year]
        
        # Reproduce original sample indices (from first script with seed 42)
        np.random.seed(42)
        original_sample_size = min(len(year_data), 20000)
        original_indices = np.random.choice(len(year_data), original_sample_size, replace=False)
        original_indices_set = set(original_indices)
        
        # Get remaining indices
        all_indices = set(range(len(year_data)))
        remaining_indices = list(all_indices - original_indices_set)
        
        # Sample 7k from remaining
        if len(remaining_indices) >= samples_per_year:
            np.random.seed(43)  # Different seed for complementary sample
            sampled_indices = np.random.choice(remaining_indices, samples_per_year, replace=False)
            complementary_data[year] = [year_data[i] for i in sampled_indices]
        else:
            print(f"Warning: Only {len(remaining_indices)} remaining samples for {year}, using all")
            complementary_data[year] = [year_data[i] for i in remaining_indices]
    
    # Convert to instruction format
    print("Converting to instruction format...")
    instruction_data = []
    
    for year in tqdm(years, desc="Creating instruction examples"):
        if year not in complementary_data:
            continue
        
        for abstract in complementary_data[year]:
            # Clean up abstract (remove extra whitespace/newlines)
            abstract = ' '.join(abstract.split())
            
            example = create_expansion_example(abstract)
            if example is not None:
                instruction_data.append(example)
    
    # Shuffle all examples
    np.random.seed(44)
    np.random.shuffle(instruction_data)
    
    print(f"\nTotal examples created: {len(instruction_data)}")

    # import pdb;pdb.set_trace()
    
    # Save in JSONL format (standard for instruction tuning)
    output_path = f"/share/garg/arxiv_kaggle/subsamples/arxiv_instruction_expansion_{category}_{samples_per_year}per_year.jsonl"
    with open(output_path, "w") as f:
        for example in instruction_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved to: {output_path}")
    
    # Also save year breakdown for reference
    year_counts = {year: len(complementary_data.get(year, [])) for year in years}
    print("\nSamples per year:")
    for year, count in sorted(year_counts.items()):
        print(f"  {year}: {count}")