# This script is to conduct VADER analysis on Reddit data

# Import necessary libraries
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

## If you haven't installed nltk ,statsmodels.api and seaborn, uncomment the following line to install it
## py -m pip install nltk statsmodels seaborn

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

# Transfer "score_submission" to log scale to reduce skewness
df['score_submission'] = df['score_submission'].apply(lambda x: x if x > 0 else 0)  # Ensure no negative values
df['log_score_submission'] = df['score_submission'].apply(lambda x: np.log1p(x))  # log1p to handle zero values

# Save the result to a new CSV file for future use
output_csv = "./25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df.to_csv(output_csv, index=False)
print("VADER analysis completed and saved to", output_csv)

# Regresion analysis related to VADER scores and the number of upvotes equivalent to a like or thumbs up("score_submission")
X = df['vader_compound']
y = df['log_score_submission']
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
print(model.summary())
# Output regression results to a text file
with open("./vader_regression_results.txt", "w") as f:
    f.write(model.summary().as_text())
print("Regression analysis completed and results saved.")

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vader_compound', y='log_score_submission', data=df)
plt.title('VADER Compound Score vs. Log of Number of Upvotes')
plt.xlabel('VADER Compound Score')
plt.ylabel('Log of Number of Upvotes')
output_png = "./vader_compound_vs_upvotes.png" # Output as a PNG file
plt.savefig(output_png)
print("Plot saved as", output_png)
plt.show()
print("Visualization completed.")
