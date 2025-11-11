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
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

def get_word_embeddings(texts, batch_size=32, max_length=128):
    all_word_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", leave=False):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        for j in range(len(batch_texts)):
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][j])
            embeddings = hidden_states[j].cpu().numpy()

            word_embs = []
            current_word = ""
            current_vecs = []

            for token, vec in zip(tokens, embeddings):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                if token.startswith("##"):
                    current_word += token[2:]
                    current_vecs.append(vec)
                else:
                    if current_word and current_vecs:
                        word_embs.append((current_word, np.mean(current_vecs, axis=0)))
                    current_word = token
                    current_vecs = [vec]

            if current_word and current_vecs:
                word_embs.append((current_word, np.mean(current_vecs, axis=0)))

            all_word_embeddings.append(dict(word_embs))

        torch.cuda.empty_cache()
    return all_word_embeddings


input_csv = "./data/reddit_sample.csv"
output_dir = "./data/reddit_word_chunks"
os.makedirs(output_dir, exist_ok=True)

chunksize = 20000
reader = pd.read_csv(input_csv, chunksize=chunksize)
global_offset = 0

for chunk_idx, chunk in enumerate(reader):
    print(f"\nProcessing chunk {chunk_idx + 1}...")
    chunk['body'] = chunk['body'].fillna('').astype(str)
    texts = chunk['body'].tolist()
    word_emb_list = get_word_embeddings(texts, batch_size=16)

    rows = [
        {"text_id": global_offset + i, "word": word, **{f"dim_{d}": emb[d] for d in range(len(emb))}}
        for i, word_dict in enumerate(word_emb_list)
        for word, emb in word_dict.items()
    ]
    df_out = pd.DataFrame(rows)

    part_path = os.path.join(output_dir, f"reddit_word_part{chunk_idx + 1}.csv")
    df_out.to_csv(part_path, index=False)
    print(f"Saved {len(df_out)} word embeddings → {part_path}")

    global_offset += len(chunk)

