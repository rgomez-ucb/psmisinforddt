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
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import statsmodels.formula.api as smf
from umap import UMAP

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load the dataset
# Use dataset after Vader sentiment analysis 
veder_csv = "./25_pct_merged_PoliticalDiscussion_comments_vader.csv"
df = pd.read_csv(veder_csv)

# Load the dataset
print("---- Loading BERTopic model. ----")
loaded_model = BERTopic.load("final_bertopic_model")

# Prepare texts for BERTopic
texts = df['body'].tolist()
print("----Finished loading dataset and preparing texts.----")

# Assign topics to the documents
print("Assigning topics to the documents...")
topics, probs = loaded_model.transform(texts) 
print("Transformed texts to topics using the loaded BERTopic model.")

# Prepare texts for BERTopic
topic_info = loaded_model.get_topic_info()
print(topic_info[['Topic', 'Representation']])
# Save topic info to CSV
topic_info.to_csv("bertopic_topic_info.csv", index=False)

# Define topic IDs for further analysis
TOPIC_ID_ABORTION = 7
TOPIC_ID_GUN_CONTROL = 9  
TOPIC_ID_INTERNATIOMAL_RELATIONS = 13

# Add topics to the original dataframe
df['topic_id'] = topics
print(f"Number of noise posts (topic ID -1): {df[df['topic_id'] == -1].shape[0]}")

# Clean the dataframe by removing noise posts
df_clean = df[df['topic_id'] != -1].copy()

# One-hot encode the topic IDs
df_reg = pd.get_dummies(df_clean, columns=['topic_id'], prefix='Topic', drop_first=True)

# Translate upvotes to log scale
df_reg['log_upvotes'] = np.log1p(df_reg['score_submission'])

# Identify topic columns
topic_cols = [col for col in df_reg.columns if col.startswith('Topic_')]
# Make formula for regression
formula = 'log_upvotes ~ ' + ' + '.join(topic_cols)
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

# Futhure analysis : Interaction regression with sentiment("vader_compound")
import statsmodels.formula.api as smf
import pandas as pd

def run_interaction_regression_with_sample(df_data, target_topic_id, 
                                           interaction_with='vader_compound', 
                                           outcome='log_upvotes',
                                           text_column='body',  
                                           sample_size=3):     
    topic_variable = f'Topic_{target_topic_id}'
    # Check if the topic variable exists in the dataframe
    if topic_variable not in df_data.columns:
        print(f"Error: Topic ID {target_topic_id} is not found as a dummy variable ({topic_variable}).")
        return None
    # Extract posts belonging to the specified topic
    df_topic_posts = df_data[df_data[topic_variable] == 1].copy()
    
    print(f"\n--- Sample posts for Topic {target_topic_id} ({topic_variable}) ---")
    print(f"Total posts in topic: {len(df_topic_posts)}")

    # Extract and display posts, VADER scores, and Upvotes
    if not df_topic_posts.empty:
        sample_posts = df_topic_posts[[text_column, 'vader_compound', outcome]].sample(min(sample_size, len(df_topic_posts)), random_state=42)
        
        for i, row in sample_posts.iterrows():
            print(f"\n[Post {i+1} Summary]")
            print(f"  VADER Compound: {row['vader_compound']:.4f}")
            print(f"  Log Upvotes: {row[outcome]:.4f}")
            print(f"  Text: {row[text_column][:100]}...") 
    else:
        print("No matching posts found.")
        
    # Run interaction regression
    base_predictors = [interaction_with]
    topic_cols = [col for col in df_data.columns if col.startswith('Topic_')]
    interaction_term = f'{interaction_with}:{topic_variable}'

    formula = f'{outcome} ~ {" + ".join(base_predictors)} + {" + ".join(topic_cols)} + {interaction_term}' 
    print(f"\n--- Running Model for Topic {target_topic_id} ---")
    print(f"Formula: {formula}")
    model = smf.ols(formula=formula, data=df_data).fit() 
    
    return model

# Run interaction regression for a specific topic 7
model_abortion = run_interaction_regression_with_sample(df_reg, target_topic_id=TOPIC_ID_ABORTION)
print(model_abortion.summary())
# Save regression results to text file
with open("bertopic_interaction_regression_topic_7_summary.txt", "w") as f:
    f.write(model_abortion.summary().as_text())

# Run interaction regression for a specific topic 9
model_gun_control = run_interaction_regression_with_sample(df_reg, target_topic_id=TOPIC_ID_GUN_CONTROL)
print(model_gun_control.summary())
# Save regression results to text file
with open("bertopic_interaction_regression_topic_9_summary.txt", "w") as f:
    f.write(model_gun_control.summary().as_text())

# Run interaction regression for a specific topic 13
model_international_relations = run_interaction_regression_with_sample(df_reg, target_topic_id=TOPIC_ID_INTERNATIOMAL_RELATIONS)
print(model_international_relations.summary())
# Save regression results to text file
with open("bertopic_interaction_regression_topic_13_summary.txt", "w") as f:
    f.write(model_international_relations.summary().as_text())

