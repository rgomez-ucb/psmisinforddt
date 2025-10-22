import pandas as pd

jsonl_file = "C:/Users/16082/Desktop/class_project/PoliticalDiscussion_cleaned.jsonl"
csv_file = "reddit_comments.csv"


chunksize = 100_000


reader = pd.read_json(jsonl_file, lines=True, chunksize=chunksize)

first_chunk = True
for chunk in reader:

    chunk["created_utc"] = pd.to_datetime(chunk["created_utc"], unit="s", errors="ignore")


    chunk = chunk[["created_utc", "score", "body","link_id"]]


    chunk.to_csv(csv_file, mode='a', index=False, header=first_chunk, encoding="utf-8-sig")
    first_chunk = False
print("finished")