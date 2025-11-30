import os
import pandas as pd
import ollama

# Load data
input_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(input_csv)
print(df.info())
# Ollama client
client = ollama.Client(host='http://127.0.0.1:11434')

# --- Combined Sentiment Function ---
def ollama_sentiment(text):
    prompt = f"""
    Analyze the sentiment of the following text.

    1. Give a sentiment label: Positive, Negative, or Neutral.
    2. Provide a sentiment score between -1 (very negative) and 1 (very positive).
    3. Output strictly in the format:
    LABEL: <label>
    SCORE: <score>

    Text: {text}
    """

    response = client.chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    content = response["message"]["content"]

    # --- Extract fields ---
    label = "Neutral"
    score = 0.0

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


# Apply to first 10 rows
df[["llm_label", "llm_score"]] = df["body"].head(10).apply(ollama_sentiment)

# Show mean score
print("Mean LLM Score (first 10 rows):", df["llm_score"].mean())

# Save output
output_csv = "./data/25_pct_merged_PoliticalDiscussion_comments_llm.csv"
df.to_csv(output_csv, index=False)
print("LLM analysis completed and saved to", output_csv)
