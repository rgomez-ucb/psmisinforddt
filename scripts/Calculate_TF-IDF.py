# This script is to calculate TF-IDF for text data

# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load the dataset
input_csv = "reddit_sample.csv"
df = pd.read_csv(input_csv)

# Initialize the TF-IDF Vectorizer
# To tackle MemoryError, limit max features and set min_df and max_df
vectorizer = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.8, stop_words='english') # change parameters as needed

# Fit and transform the 'body' column
# First, Null in "body" change to empty string
df['body'] = df['body'].fillna("") 
tfidf_matrix = vectorizer.fit_transform(df['body'])
# To reduce memory usage, convert to sparse DataFrame
tfidf_df_sparse = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vectorizer.get_feature_names_out())

# Combine TF-IDF features with the original dataframe
result_df = pd.concat([df.reset_index(drop=True), tfidf_df_sparse.reset_index(drop=True)], axis=1)

# Save the result to a new CSV file
output_csv = "reddit_tfidf.csv"
result_df.to_csv(output_csv, index=False)
print("TF-IDF calculation completed and saved to", output_csv)