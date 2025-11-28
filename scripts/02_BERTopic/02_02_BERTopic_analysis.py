# This script is to conduct BERTopic analysis on Reddit data

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
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load the dataset
# Use dataset after adding BERTopic topic assignments
bertopic_csv = "./PoliticalDiscussion_with_vader_bertopic.csv"
df = pd.read_csv(bertopic_csv)

# Clean the dataframe by removing noise posts
print(f"Number of noise posts (topic ID -1): {df[df['topic_id'] == -1].shape[0]}")
df_clean = df[df['topic_id'] != -1].copy()

# Save the cleaned dataframe with topics to CSV
df_clean.to_csv("bertopic_assigned_posts_cleaned.csv", index=False)

# One-hot encode the topic IDs
df_reg = pd.get_dummies(df_clean, columns=['topic_id'], prefix='Topic', drop_first=True)

# Translate upvotes to log scale
df_reg['log_upvotes'] = np.log1p(df_reg['score_submission'])

# Identify topic columns
topic_cols = [col for col in df_reg.columns if col.startswith('Topic_')]
# Make formula for regression
formula = 'log_upvotes ~ vader_compound + ' + ' + '.join(topic_cols)
print("\n--- Regression Formula ---")
print(formula)
# Execute regression
model = smf.ols(formula=formula, data=df_reg).fit()
# Print regression summary
print("\n--- Regression Summary ---")
print(model.summary())
# Save regression results to text file
with open("bertopic_regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())
# Save the dataframe with topics to CSV
df_reg.to_csv("bertopic_regression_data.csv", index=False)

# Extract regression coefficients and p-values
results_table = model.summary2().tables[1]

# Save the results table with p-values to CSV
results_table.to_csv("regression_coefficients_with_pvalue.csv", index=True)

print("✅ Regression results with p-values have been saved to CSV.")