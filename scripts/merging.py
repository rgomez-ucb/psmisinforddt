import pandas as pd

submissions = pd.read_csv("./data/reddit_submissions.csv", low_memory=False)



sampled_comments = pd.read_csv("./data/reddit_comments_sample.csv", low_memory=False)
sampled_comments["submission_id"] = sampled_comments["link_id"].str.replace(r"t[0-9]_", "", regex=True)
submission_ids = sampled_comments["submission_id"].unique()

submission_ids = sampled_comments["submission_id"].unique()

# Submissions that match comments
matched_submissions = submissions[submissions["id"].isin(submission_ids)]

# Submissions not in sampled comments → we sample 30%
remaining = submissions[~submissions["id"].isin(submission_ids)]
remaining_sample = remaining.sample(frac=0.30, random_state=6657)

# Combine matched + 30% unmatched
final_submissions = pd.concat([matched_submissions, remaining_sample], ignore_index=True)


merged = final_submissions.merge(
    sampled_comments,
    left_on="id",
    right_on="submission_id",
    how="left"           
)

# Save result
merged.to_csv("./data/merged.csv", index=False, encoding="utf-8-sig")

print("finished")