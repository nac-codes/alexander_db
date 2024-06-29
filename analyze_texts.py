# analyze_texts.py

import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from scipy import stats
import matplotlib.pyplot as plt
import argparse
import json

try:
    import torch
    from transformers import BertTokenizer, BertModel
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def get_embedding(text, model_name, tokenizer=None, model=None, client=None):
    if model_name == "bert":
        if not BERT_AVAILABLE:
            raise ImportError("BERT model is not available. Please install transformers and torch.")
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()
    elif model_name == "openai-gpt":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI GPT is not available. Please install openai.")
        text = text.replace("\n", " ")
        try:
            response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
            return np.array(response.data[0].embedding)
        except openai.APIError as e:
            print(f"Error when processing text: {text}...")  # Print the first 50 characters of the problematic text
            print(f"OpenAI API returned an API Error: {e}")
            return np.zeros(1536)
    else:
        raise ValueError("Invalid model name")

def get_ngrams(text, n):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def multi_ngram_search(query, text, max_n=3):
    query_ngrams = [get_ngrams(query, i) for i in range(1, max_n+1)]
    text_ngrams = [get_ngrams(text, i) for i in range(1, max_n+1)]
    
    score = 0
    max_score = 0  # Initialize max_score to calculate the maximum possible score
    
    # Calculate score
    for n in range(1, max_n+1):
        matches = set(query_ngrams[n-1]) & set(text_ngrams[n-1])
        score += len(matches) * n
        max_score += len(query_ngrams[n-1]) * n  # Accumulate maximum possible score
    
    # Normalize score
    if max_score > 0:
        normalized_score = score / max_score
    else:
        normalized_score = 0.0  # Handle the case where max_score is 0 (division by zero)
    
    return normalized_score

def search_chunks(chunk_data, chunk_embeddings, query, model_name, cosine_weight, tokenizer=None, model=None, client=None):
    query_embedding = get_embedding(query, model_name, tokenizer, model, client)
    
    results = []
    for chunk, chunk_embedding in enumerate(chunk_embeddings):
        cosine_similarity = 1 - cosine(query_embedding, chunk_embedding)
        ngram_score = multi_ngram_search(query, chunk_data[chunk]['content'])
        
        # Combine scores using the specified weighting
        combined_score = (cosine_similarity * cosine_weight) + (ngram_score * (1 - cosine_weight))
        
        results.append((chunk_data[chunk], combined_score))
    
    return sorted(results, key=lambda x: x[1], reverse=True)

def create_distribution_graph(results, title_suffix="", filename_suffix=""):
    title_scores = defaultdict(lambda: {'greek': [], 'translation': []})
    for chunk, score in results:
        if chunk['is_greek']:
            title_scores[chunk['title']]['greek'].append(score)
        else:
            title_scores[chunk['title']]['translation'].append(score)
    
    plt.figure(figsize=(12, 6))
    
    for title, scores in title_scores.items():
        for lang, lang_scores in scores.items():
            if lang_scores:
                kde = stats.gaussian_kde(lang_scores)
                x_range = np.linspace(min(lang_scores), max(lang_scores), 200)
                plt.plot(x_range, kde(x_range), label=f"{title} ({lang.capitalize()})")
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title(f'Distribution of Similarity Scores by Title and Language {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'similarity_distribution{filename_suffix}.jpg', dpi=300, bbox_inches='tight')
    plt.close()

    return title_scores

def calculate_statistics(scores):
    return {
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std_dev': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores)
    }

def print_statistics(title_scores, description=""):
    print(f"\n--- Statistics for {description} ---")
    for title, scores in title_scores.items():
        print(f"\n{title}:")
        for lang in ['greek', 'translation']:
            if scores[lang]:
                stats = calculate_statistics(scores[lang])
                print(f"  {lang.capitalize()}:")
                print(f"    Mean: {stats['mean']:.4f}")
                print(f"    Median: {stats['median']:.4f}")
                print(f"    Standard Deviation: {stats['std_dev']:.4f}")
                print(f"    Min: {stats['min']:.4f}")
                print(f"    Max: {stats['max']:.4f}")
            else:
                print(f"  {lang.capitalize()}: No data")

def save_results_to_json(results, model_name):
    json_results = []
    for chunk, score in results:
        result = {
            "score": score,
            "title": chunk['title'],
            "page": chunk['page'],
            "is_greek": chunk['is_greek'],
            "content": chunk['content']
        }
        if 'volume' in chunk:
            result['volume'] = chunk['volume']
        if 'book' in chunk:
            result['book'] = chunk['book']
        json_results.append(result)
    
    with open(f'results_{model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Analyze texts using embeddings")
    parser.add_argument("--model", choices=["bert", "openai-gpt"], default="bert", help="Choose the model for analysis")
    args = parser.parse_args()

    model_name = args.model
    chunk_data_file = f'chunks_txt_{model_name}.npy'
    chunk_embeddings_file = f'chunks_embeddings_{model_name}.npy'

    print("Loading chunk data and embeddings...")
    chunk_data = np.load(chunk_data_file, allow_pickle=True)
    chunk_embeddings = np.load(chunk_embeddings_file)
    tokenizer = None
    model = None
    client = None

    if model_name == "bert":
        if not BERT_AVAILABLE:
            print("Error: BERT model is not available. Please install transformers and torch.")
            return
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif model_name == "openai-gpt":
        if not OPENAI_AVAILABLE:
            print("Error: OpenAI GPT is not available. Please install openai.")
            return
        client = OpenAI()

    print("Enter your search query:")
    query = input()
    
    print("Enter the weight for cosine similarity (0 to 1, where 0.5 is equal weighting):")
    cosine_weight = float(input())
    
    print("Enter the similarity threshold (0 to 1):")
    similarity_threshold = float(input())
    
    try:
        all_results = search_chunks(chunk_data, chunk_embeddings, query, model_name, cosine_weight, tokenizer, model, client)
    except ImportError as e:
        print(f"Error: {str(e)}")
        return
    
    # Process all results
    all_title_scores = create_distribution_graph(all_results, "(All Results)", "_all")
    print_statistics(all_title_scores, "All Results")
    
    # Process results above threshold
    filtered_results = [(chunk, score) for chunk, score in all_results if score >= similarity_threshold]
    filtered_title_scores = create_distribution_graph(filtered_results, f"(Similarity >= {similarity_threshold})", "_filtered")
    print_statistics(filtered_title_scores, f"Results with Similarity >= {similarity_threshold}")
    
    print("\nTop 10 results:")
    for i, (chunk, score) in enumerate(filtered_results[:10], 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"Title: {chunk['title']}")
        if 'volume' in chunk:
            print(f"Volume: {chunk['volume']}")
        if 'book' in chunk:
            print(f"Book: {chunk['book']}")
        print(f"Page: {chunk['page']}")
        print(f"Is Greek: {chunk['is_greek']}")
        # clean up content
        content = chunk['content'].replace("\n", " ").strip()
        print(f"Content: {content}...")

    # Save results to JSON
    save_results_to_json(filtered_results, model_name)
    print(f"\nAll results saved to results_{model_name}.json")

if __name__ == "__main__":
    main()