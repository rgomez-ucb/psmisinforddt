# This script is to conduct VADER analysis on Reddit data

# Import necessary libraries
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

## If you haven't installed nltk, uncomment the following line to install it
## py -m pip install nltk

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load the dataset
input_csv = "./25_pct_merged_PoliticalDiscussion_comments.csv"
df = pd.read_csv(input_csv)

# VADER Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Apply VADER to the 'body' column
df['vader_scores'] = df['body'].apply(lambda x: sia.polarity_scores(str(x)))

# Extract compound score
df['vader_compound'] = df['vader_scores'].apply(lambda score_dict: score_dict['compound'])

# Print something :)
mean_vader_compound = df['vader_compound'].mean()
print("Mean VADER Compound Score:", mean_vader_compound)

# Save the result to a new CSV file
output_csv = "./25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df.to_csv(output_csv, index=False)
print("VADER analysis completed and saved to", output_csv)

# Clasify sentiment based on compound score
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
df['vader_sentiment'] = df['vader_compound'].apply(classify_sentiment)
print("Sentiment classification added.")

# Display the first few rows of the updated DataFrame
print(df.head())

