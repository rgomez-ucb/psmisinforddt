# 30% OF THE DATASET
# CHANGE THE FILE PATH
import pandas as pd
import numpy as np
input_csv = "reddit_comments.csv"
output_csv = "reddit_sample.csv"
df = pd.read_csv(input_csv, parse_dates=["created_utc"])
df["year"] = df["created_utc"].dt.year
sampled_df = df.groupby("year", group_keys=False).apply(
    lambda x: x.sample(frac=0.3, random_state=6657)
)
sampled_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print("finsihed")
