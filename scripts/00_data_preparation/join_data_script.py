import os
import pandas as pd

comments_polidisc = ".././data/PoliticalDiscussion_comments_sample.csv"
submissions_polidisc = ".././data/PoliticalDiscussion_submissions_sample.csv"

merged_comments_submissions_politicaldiscussion = ".././data/merged_PoliticalDiscussion_comments_submissions.csv"
sample_comments_submissions_politicaldiscussion = ".././data/25_pct_merged_PoliticalDiscussion_comments_submissions_merged_25pct.csv"
print("Reading CSV files...")
comments_df = pd.read_csv(comments_polidisc)
submissions_df = pd.read_csv(submissions_polidisc)

print("Comments shape:", comments_df.shape)
print("Submissions shape:", submissions_df.shape)

print("merging on 'submission_id'...")
merged_df = pd.merge(
    comments_df, 
    submissions_df, 
    on='submission_id', 
    how='inner',
    suffixes=('_comment', '_submission')
)

print("Merged shape:", merged_df.shape)

print("Saving merged data to CSV...")
merged_df.to_csv(merged_comments_submissions_politicaldiscussion, index=False)

Sample_fraction = 0.25

print(f"Sampling {Sample_fraction*100: .0f}% of the merged data...")
sampled_df = merged_df.sample(frac=Sample_fraction, random_state=42)

print("Sampled shape:", sampled_df.shape)
print("Saving sampled data to CSV...")
sampled_df.to_csv(sample_comments_submissions_politicaldiscussion, index=False)

def print_file_size(path):
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"{path}: {size_mb:.2f} MB")

print("File sizes:")
print_file_size(comments_polidisc)
print_file_size(submissions_polidisc)
print_file_size(merged_comments_submissions_politicaldiscussion)
print_file_size(sample_comments_submissions_politicaldiscussion)
print("Done.")