import ollama
import pandas as pd
from tqdm import tqdm
import ast  # for safe literal parsing

client = ollama.Client(host='http://127.0.0.1:11434')

# Load your existing dataset
input_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(input_csv)

def qwen_label(text):

    prompt = f"""
You are a labeling model.

TASK 1 — MISINFORMATION (0/1):
0 = not misinformation  
1 = misinformation  

TASK 2 — TOKEN LABELING:
For each token:
O = objective  
N = narrative / opinion / emotional  

TASK 3 — NARRATIVE SCORE:
narrative_score = (# of N tokens) / (total tokens)

Return output ONLY in the following format:

MISINFO: <0 or 1>
NARRATIVE_SCORE: <0.0-1.0>
TOKENS: token1, token2, token3
LABELS: O, N, O

Text:
{text}
"""

    response = client.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={
            "temperature": 0,
            "max_tokens": 200,
            "top_p": 0.9,
        }
    )

    raw = response["message"]["content"].strip()

    # ---- Parse result ----
    misinfo, score, tokens, labels = None, None, [], []

    for line in raw.splitlines():
        if line.startswith("MISINFO:"):
            misinfo = line.replace("MISINFO:", "").strip()
        elif line.startswith("NARRATIVE_SCORE:"):
            score = line.replace("NARRATIVE_SCORE:", "").strip()
        elif line.startswith("TOKENS:"):
            tokens_str = line.replace("TOKENS:", "").strip()
            tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]
        elif line.startswith("LABELS:"):
            labels_str = line.replace("LABELS:", "").strip()
            labels = [t.strip() for t in labels_str.split(",") if t.strip()]

    # Convert types safely
    try:
        misinfo = int(misinfo)
    except:
        misinfo = None

    try:
        score = float(score)
    except:
        score = None

    return {
        "misinfo_label": misinfo,
        "narrative_score": score,
        "tokens": tokens,
        "labels": labels,
        "raw": raw
    }


# Prepare empty columns
df["misinfo_label"] = None
df["narrative_score"] = None
df["tokens_labels"] = None

# Loop through dataset
for i, text in tqdm(df["body"].items(), total=len(df)):
    result = qwen_label(text)

    df.at[i, "misinfo_label"] = result["misinfo_label"]
    df.at[i, "narrative_score"] = result["narrative_score"]
    df.at[i, "tokens_labels"] = str({
        "tokens": result["tokens"],
        "labels": result["labels"]
    })

# Save updated dataset
output_csv = "./data/llm/PoliticalDiscussion_with_llm_labels.csv"
df.to_csv(output_csv, index=False)

print(f"Saved annotated dataset → {output_csv}")
