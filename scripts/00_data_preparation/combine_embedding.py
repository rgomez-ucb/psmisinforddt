import os
import pandas as pd
output_dir = ".././data/reddit_bert_chunks"
final_csv = ".././data/reddit_embeddings_all.csv"

all_parts = [
    os.path.join(output_dir, f)
    for f in sorted(os.listdir(output_dir))
    if f.endswith(".csv")
]


if os.path.exists(final_csv):
    os.remove(final_csv)

first = True
for f in all_parts:
    print(f"Merging {f} ...")
    chunk = pd.read_csv(f)
    chunk.to_csv(final_csv, mode="a", index=False, header=first)
    first = False

