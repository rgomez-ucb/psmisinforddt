# This script is to conduct VADER analysis on Reddit data

# Import necessary libraries
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load the dataset
input_csv = "./data/reddit_sample.csv"
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
output_csv = "./data/reddit_vader.csv"
df.to_csv(output_csv, index=False)
print("VADER analysis completed and saved to", output_csv)