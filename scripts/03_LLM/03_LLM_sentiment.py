import os
import pandas as pd
import ollama

input_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(input_csv)

client = ollama.Client(host='http://127.0.0.1:11434')

def ollama_sentiment(text):
    prompt = f"""
    Classify the sentiment as Positive, Negative, or Neutral.
    Give a score between -1 and 1.
    Format:
    LABEL: <label>
    SCORE: <score>

    Text: {text}
    """

    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": 30,    
            "temperature": 0.0, 
        },
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
    if i % 10 == 0:
        print(f"Processing row {i}...")

    results.append(ollama_sentiment(text))


df[["llm_label", "llm_score"]] = pd.DataFrame(results)

output_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_llm.csv"
df.to_csv(output_csv, index=False)

print("\nDONE. Saved to:", output_csv)
