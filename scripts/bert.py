import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Dev1ce:", device)


model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()


def get_bert_embeddings(texts, batch_size=64):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", leave=False):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype('float32')
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)


input_csv = "./data/reddit_sample.csv"
output_dir = "./data/reddit_bert_chunks"
final_csv = "./data/reddit_distilbert_embeddings.csv"
os.makedirs(output_dir, exist_ok=True)

chunksize = 50000 
reader = pd.read_csv(input_csv, chunksize=chunksize)

for chunk_idx, chunk in enumerate(reader):
    print(f"\n Processing chunk {chunk_idx+1}...")
    chunk['body'] = chunk['body'].fillna('').astype(str)
    texts = chunk['body'].tolist()

    embeddings = get_bert_embeddings(texts, batch_size=64)
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])

    merged = pd.concat([chunk.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


    part_path = os.path.join(output_dir, f"reddit_embeddings_part{chunk_idx+1}.csv")
    merged.to_csv(part_path, index=False)
    print(f"Saved {len(chunk)} rows → {part_path}")

    torch.cuda.empty_cache()

# careful !!! if memory issue, run another code. You have saved all the parts already.
all_parts = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir)) if f.endswith(".csv")]
combined_df = pd.concat((pd.read_csv(f) for f in all_parts), ignore_index=True)
combined_df.to_csv(final_csv, index=False)
