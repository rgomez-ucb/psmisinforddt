# 30% OF THE DATASET
# CHANGE THE FILE PATH
import pandas as pd
import numpy as np
input_csv = ".././data/reddit_comments.csv"
output_csv = ".././data/reddit_comments_sample.csv"
df = pd.read_csv(input_csv,low_memory=False)
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
df["year"] = df["created_utc"].dt.year
sampled_df = df.groupby("year", group_keys=False).apply(
    lambda x: x.sample(frac=0.15, random_state=6657)
)
sampled_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print("finsihed")
