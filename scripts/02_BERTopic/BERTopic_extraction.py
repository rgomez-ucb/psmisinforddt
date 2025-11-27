# This script is to conduct BERTopic extraction on Reddit data

# If you haven't installed below libralies, uncomment the following line to install them
# pip install bertopic sentence-transformers umap-learn hdbscan

import sys
print("--- print the path ---")
print(sys.executable)
print("-------------------------------------")

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import statsmodels.formula.api as smf

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

# Initialize BERTopic with custom parameters
topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",        
    calculate_probabilities=True,
    verbose=True,
    min_topic_size=50,
    umap_model=umap_model
)

# Extract topics
print("Extracting topics...")
topics, probs = topic_model.fit_transform(texts)
print("Topic extraction completed.")

topic_model.save("final_bertopic_model")
print("Saved BERTopic model as 'final_bertopic_model'.")