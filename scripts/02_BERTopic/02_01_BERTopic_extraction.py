# This script is to conduct BERTopic extraction on Reddit data

# If you haven't installed below libralies, uncomment the following line to install them
# pip install bertopic sentence-transformers umap-learn hdbscan

import sys
print("--- print the path ---")
print(sys.executable)
print("-------------------------------------")

# Import necessary libraries
import os
import nltk
import pandas as pd
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import statsmodels.formula.api as smf
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load the dataset
# Use dataset after Vader sentiment analysis
veder_csv = "./25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(veder_csv)

# Prepare texts for BERTopic
texts = df['body'].tolist()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# random_state for reproducibility
umap_model = UMAP(random_state=42)

# Create stop words to HDBSCAN
english_stopwords = stopwords.words('english')
# Add custom noise words relevant to Reddit political discussions
custom_noise = [
    # Names of politicians
    'trump', 'biden', 
    # Reddit-specific noise
    'reddit', 'user', 'post', 'comment', 'thread', 
    # Opinion expression/general political terms
    'politic', 'government', 'state', 'country', 'said', 'agree', 'believe', 'opinion', 
    # Common verbs/pronouns
    'say', 'think', 'just', 'like', 'know', 'people', 
]
custom_stopwords_list = english_stopwords + custom_noise

# Define a Vectorizer model with applied stop words
vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords_list 
)

# Initialize BERTopic with custom parameters
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,        
    calculate_probabilities=True,
    verbose=True,
    min_topic_size=50,
    umap_model=umap_model
)

# Extract topics
print("\n--- Starts to extract topic assignments and save model ---")
topics, probs = topic_model.fit_transform(texts)
print("Topic extraction completed.")

# Assign topics to the original dataframe
df['topic_id'] = topics

# Save the BERTopic model
MODEL_SAVE_PATH = "final_bertopic_model"
topic_model.save(MODEL_SAVE_PATH)
print(f"✅ Saved BERTopic model successfully to: '{MODEL_SAVE_PATH}'")

# Save the dataframe with topics to CSV
DATA_SAVE_PATH = "PoliticalDiscussion_with_vader_bertopic.csv"
df.to_csv(DATA_SAVE_PATH, index=False)
print(f"✅ Final data with topic IDs saved to: '{DATA_SAVE_PATH}'")

# Extract topic information such as ID, size, and keywords as a DataFrame
try:
    topic_model = BERTopic.load(MODEL_SAVE_PATH)
    print("✅ Successfully loaded BERTopic model for topic info extraction.")
    OUTPUT_CSV_PATH = "bertopic_topic_info.csv"
    topic_info_df = topic_model.get_topic_info()
    print(f"✅ Number of topics: {len(topic_info_df)}")
    # Save topic information to CSV
    topic_info_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Successfully saved topic information table to CSV: {OUTPUT_CSV_PATH}")
except Exception as e:
    print(f"❌ Failed to extract BERTopic information. Error: {e}")