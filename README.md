# COMPSS-211-Final-Project_Political_misinformation

This repository facilitates collaborative efforts to analyze political misinformation using Reddit data, as part of the Advanced Computing course within the MaCSS program.

## Group members
Shun, Ruben, Jiyang and Yucheng.

### Role
- Jiyang Li : NLP lead
- Shun Moriguchi : Data Analyst Lead
- Juan Ruben Gomez : Exploratory research lead
- Yucheng Lu : Visualization lead

## Information
This is the repository where we should work!  
Our presentation day is December 1st.

## Research Question
How does misinformation circulate and engage communities across political subreddits on Reddit over time?

## Dataset
Reddit  r/PoliticalDiscussion

## What we did and plan to do
1. TF-IDF
2. BERT topic: to categorize the posts and the comments.
3.  Get the top posts then run VADER analysis  
    - Do negative or positive posts get more feedback?
4. Comparison models: Linear regression, SVM, RNN, CNN
5. F1 score, confusion matrix, 

## Project Structure

This project is structured with the following directory layout:

```
|   .DS_Store
|   .gitignore
|   environment.yml
|   political-misinfo-reddit-data.toml
|   README.md
|   repository_structure.txt  # For making this tree
|   
+---.devcontainer
|       devcontainer.json
|       
+---.vscode
|       settings.json
|       
+---data
|       .DS_Store
|       .gitkeep
|       don_trump_highlighted.png
|       joined_data.csv
|       people_highlighted_term.png
|       PoliticalDiscussion_comments_sample.csv
|       PoliticalDiscussion_submissions_sample.csv
|       reddit-1614740ac8c94505e4ecb9d88be8bed7b6afddd4.torrent
|       reddit_politicaldisscussion_linechart.html
|       reddit_politicaldisscussion_linechart.Rmd
|       reddit_rpoliticaldiscussion_linechart.R
|       reddit_rPoliticalDiscussion_YoY.jpeg
|       Rplot.jpeg
|       Rplot01.jpeg
|       tf-idf-score.png
|       tfidf_top20_interactive.html
|       tfidf_viz.ipynb
|       word2vec_clusters.png
|       
+---notebooks
|       .gitkeep
|       presentation 2 codes.ipynb
|       presentation 2.ipynb
|       project_test.db
|       
+---scripts
|   |   .gitkeep
|   |   bert.py
|   |   Calculate_TF-IDF.py
|   |   cluster.py
|   |   combine_embedding.py
|   |   random_split.py
|   |   test_0.py
|   |   to_csv.py
|   |   word2vec.py
|   |   
|   \---Practice   # Old files
|           
\---src
    |   .DS_Store
    |   data_pipeline.py
    |   
    +---reddit_data
    |   |   mymodule.py
    |   |   tfidf_top20_interactive.html
    |   |   utils.py
    |   |   __init__.py
    |   |   
    |   \---__pycache__
    |           utils.cpython-313.pyc
    |           
    \---__pycache__
            __init__.cpython-313.pyc
```