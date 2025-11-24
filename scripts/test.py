import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
## If you haven't installed nltk ,statsmodels.api and seaborn, uncomment the following line to install it
## py -m pip install nltk statsmodels seaborn


input_csv = "./data/25_pct_merged_PoliticalDiscussion_comments_submissions_merged_25pct.csv"
df = pd.read_csv(input_csv)
# have ollama standby at port 11434
# change the model if you want
client = ollama.Client(host='http://127.0.0.1:11434')
def ollama_sentiment_analysis(text):
    prompt = f'''
    Analyze the sentiment of the following text and classify it as Positive, Negative or Neutral.'''
    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    content = response['message']['content']
    return content

# Apply LLM to the 'body' column
df['llm_class'] = df['body'].head(10).apply(lambda x: ollama_sentiment_analysis(x))


def ollama_sentiment_score(text):
    prompt = f'''
    Analyze the sentiment of the following text and provide a sentiment score between -1 (very negative) to 1 (very positive).'''
    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    content = response['message']['content']
    try:
        score = float(content.strip())
    except ValueError:
        score = 0.0  # Default to neutral if parsing fails
    return score
# Calculate llm score
df['llm_score'] = df['body'].head(10).apply(ollama_sentiment_score)

# Print something :)
mean_llm_score = df['llm_score'].mean()
print("Mean LLM Score:", mean_llm_score)

# Save the result to a new CSV file
output_csv = "./25_pct_merged_PoliticalDiscussion_comments_llm.csv"
df.to_csv(output_csv, index=False)
print("LLM analysis completed and saved to", output_csv)



