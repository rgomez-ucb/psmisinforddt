# This script is to calculate TF-IDF for text data

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import numpy as np
import re

'''
# Set working directory
 current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)
'''
# Load the dataset
input_csv = "./data/reddit_sample.csv"
df = pd.read_csv(input_csv)

# Initialize the TF-IDF Vectorizer
# To tackle MemoryError, limit max features and set min_df and max_df
# increase max feature to 10000, it works for me 
vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8, stop_words='english') # change parameters as needed

# Fit and transform the 'body' column
# First, Null in "body" change to empty string

# preprocessing
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'http\S+|www\.\S+', ' ', text)        
    text = re.sub(r'u\/\w+|r\/\w+', ' ', text)           
    text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)          
    text = re.sub(r'[^a-z\s]', ' ', text)                
    text = re.sub(r'\s+', ' ', text).strip()            
    return text

df['body'] = df['body'].apply(clean_text)

df['body'] = df['body'].fillna("")
tfidf_matrix = vectorizer.fit_transform(df['body'])
# To reduce memory usage, convert to sparse DataFrame
tfidf_df_sparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vectorizer.get_feature_names_out())

# Combine TF-IDF features with the original dataframe
result_df = pd.concat([df.reset_index(drop=True), tfidf_df_sparse.reset_index(drop=True)], axis=1)


# print something :)
mean_tfidf = tfidf_df_sparse.mean(axis=0).sort_values(ascending=False)
print("Top 20 words by TF-IDF:")
print(mean_tfidf.head(20))

# Save the result to a new CSV file
output_csv = "./data/reddit_tfidf.csv"
result_df.to_csv(output_csv, index=False)
print("TF-IDF calculation completed and saved to", output_csv)