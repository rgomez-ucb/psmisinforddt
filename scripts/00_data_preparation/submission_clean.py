import json
#change the path to your own path
#get the data from using the torrent
input_path = ".././data/PoliticalDiscussion_submissions"
output_path = ".././data/PoliticalDiscussion_cleaned_submissions.jsonl"

# thing we want to keep
fields = ["id", "created_utc", "title", "selftext", "ups","downs"]

import json

count_in, count_out = 0, 0

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        count_in += 1
        
        # skip empty lines
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip malformed lines


        # keep only specified fields
        clean_data = {k: data.get(k) for k in fields}

        try:
            fout.write(json.dumps(clean_data, ensure_ascii=False) + "\n")
            count_out += 1
        except Exception as e:
            print("Write error:", e)
            continue

print(f"Input lines: {count_in:,}，Output lines: {count_out:,}")