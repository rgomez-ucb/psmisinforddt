import pandas as pd
# CHANGE FILEPATH
jsonl_file = ".././data/PoliticalDiscussion_cleaned_submissions.jsonl"
csv_file = ".././data/reddit_submissions.csv"


chunksize = 100000


reader = pd.read_json(jsonl_file, lines=True, chunksize=chunksize)

first_chunk = True
for chunk in reader:

    chunk["created_utc"] = pd.to_datetime(chunk["created_utc"], unit="s")


    chunk = chunk[["created_utc", "ups", "selftext","id","title","downs"]]


    chunk.to_csv(csv_file, mode='a', index=False, header=first_chunk)
    first_chunk = False
print("finished")