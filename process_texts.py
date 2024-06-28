# Description: This script processes the texts in the specified folder and creates embeddings for each chunk of text.

import os
import re
import numpy as np
from tqdm import tqdm
import argparse

# Initialize OpenAI client
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
            return None
    else:
        raise ValueError("Invalid model name")

def extract_greek(text):
    greek_pattern = re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]+')
    return ' '.join(greek_pattern.findall(text))

def process_files(folder_path, model_name, tokenizer=None, model=None, client=None):
    chunk_data = []
    chunk_embeddings = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for filename in tqdm(files, desc="Processing files", unit="file"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
        
        greek_content = extract_greek(content)
        translation_content = re.sub(r'[\u0370-\u03FF\u1F00-\u1FFF]+', '', content).strip()
        if greek_content.strip() == '' or translation_content.strip() == '':
            tqdm.write(f"Skipping file {filename} with empty Greek or translation content.")
            continue

        if filename.startswith('plutarch_lives_alexanderPage_'):
            title = 'Plutarch Lives Alexander'
            page = filename.split('_')[-1].replace('.txt', '')
        elif filename.startswith('arrian_anabasispage_'):
            title = 'Arrian Anabasis'
            page = filename.split('_')[-1].replace('.txt', '')
        elif filename.startswith('DIODORUS_SICULUS_Volume_'):
            title = 'Diodorus Siculus'
            parts = filename.split('_')
            volume = parts[3]
            book = parts[5]
            page = parts[-1].replace('.txt', '')
        else:
            continue  # Skip files that don't match expected patterns
        
        # Create Greek chunk
        greek_chunk = {
            'title': title,
            'page': page,
            'content': greek_content,
            'is_greek': True
        }
        if 'volume' in locals():
            greek_chunk['volume'] = volume
            greek_chunk['book'] = book
        chunk_data.append(greek_chunk)
        chunk_embeddings.append(get_embedding(greek_content, model_name, tokenizer, model, client))
        
        # Create Translation chunk
        translation_chunk = {
            'title': title,
            'page': page,
            'content': translation_content,
            'is_greek': False
        }
        if 'volume' in locals():
            translation_chunk['volume'] = volume
            translation_chunk['book'] = book
        chunk_data.append(translation_chunk)
        chunk_embeddings.append(get_embedding(translation_content, model_name, tokenizer, model, client))

    return chunk_data, np.array(chunk_embeddings)

def main():
    parser = argparse.ArgumentParser(description="Analyze texts using embeddings")
    parser.add_argument("--model", choices=["bert", "openai-gpt"], default="bert", help="Choose the model for analysis")
    args = parser.parse_args()

    model_name = args.model
    folder_path = "./all_primaries"
    chunk_data_file = f'chunks_txt_{model_name}.npy'
    chunk_embeddings_file = f'chunks_embeddings_{model_name}.npy'

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

    print("Processing files and creating new chunk data and embeddings...")
    try:
        chunk_data, chunk_embeddings = process_files(folder_path, model_name, tokenizer, model, client)
    except ImportError as e:
        print(f"Error: {str(e)}")
        return
    
    print("Saving chunk data and embeddings...")
    np.save(chunk_data_file, chunk_data)
    np.save(chunk_embeddings_file, chunk_embeddings)
    
    print(f"Processing complete. Data saved to {chunk_data_file} and {chunk_embeddings_file}")

if __name__ == "__main__":
    main()