import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Dev1ce:", device)

model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.to(device)
model.eval()


def get_bert_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


input_csv = "./data/reddit_sample.csv"
output_csv = "./data/reddit_distilbert_embeddings.csv"

chunk_iter = pd.read_csv(input_csv, chunksize=20000)
for i, chunk in enumerate(chunk_iter):
    print(f"\nProcessing chunk {i+1}...")
    chunk['body'] = chunk['body'].fillna('').astype(str)
    texts = chunk['body'].tolist()

    embeddings = get_bert_embeddings(texts, batch_size=32)
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{j}' for j in range(embeddings.shape[1])])

    result = pd.concat([chunk.reset_index(drop=True), emb_df], axis=1)
    result.to_csv(f"reddit_embeddings_part{i+1}.csv", index=False)
    print(f"✅ Saved part {i+1} ({len(chunk)} rows)")