import os
import pandas as pd
import ollama

input_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(input_csv)

client = ollama.Client(host='http://127.0.0.1:11434')

def ollama_sentiment(text):
    prompt = f"""
    Analyze the sentiment of the following text.
    1. Give a sentiment label: Positive, Negative, or Neutral.
    2. Provide a sentiment score between -1 and 1.
    3. Output strictly:
    LABEL: <label>
    SCORE: <score>

    Text: {text}
    """

    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    content = response["message"]["content"]

    # Parse
    label, score = "Neutral", 0.0
    for line in content.split("\n"):
        line = line.strip()
        if line.lower().startswith("label:"):
            label = line.split(":", 1)[1].strip()
        if line.lower().startswith("score:"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except:
                score = 0.0

    return pd.Series([label, score])


# --- APPLY with simple progress logging ---
results = []
for i, text in df["body"].items():

    # every 100 rows, print progress
    if i % 100 == 0:
        print(f"Processing row {i}...")

    results.append(ollama_sentiment(text))


df[["llm_label", "llm_score"]] = pd.DataFrame(results)

output_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_llm.csv"
df.to_csv(output_csv, index=False)

print("\nDONE. Saved to:", output_csv)
