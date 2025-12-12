# COMPSS-211-Final-Project_Political_misinformation
This repository facilitates collaborative efforts to analyze political misinformation using Reddit data, as part of the Advanced Computing course within the MaCSS program.

## Group members
Shun, Ruben, Jiyang and Yucheng.

### Role
- **Jiyang Li** : NLP lead
- **Shun Moriguchi** : Data Analyst Lead
- **Juan Ruben Gomez** : Exploratory research lead
- **Yucheng Lu** : Visualization lead

## Information
This is the repository where we should work!  
Our final presentation day is December 1st.

## Research Question
How does misinformation circulate and engage communities across political subreddits on Reddit over time?

## Dataset
Our analysis draws on posts and comments from **r/PoliticalDiscussion**, a large Reddit community with over 2.1 million members engaged in discourse about current events, political ideology, and a range of politicized topics.  
To ensure the dataset remained manageable we extracted a 25% stratified sample of available posts and comments.

## Our analytical methods
1. **VADER Sentiment Analysis**  
To establish a baseline for our analysis, we first examined the relationship between sentiment and engagement using the VADER model. VADER was chosen for its lexicon- and rule-based approach, which is specifically tuned for sentiment analysis in social media text.
2. **Categorizing by BERTopic and executing an OLS regression**  
To quantify the contextual effect of political subreddits, we utilized the BERTopic framework to categorize post bodies into distinct political themes.
3. **Using BERT on misinformation prediction**   
    - We will first use an LLM to annotate the Political Discussion dataset and generate the variable misinfo_label to indicate whether each post contains misinformation. 
    - After that, we will split the dataset into a 4:1 training test set. Using the training set, we will train four models: a BERT model, a logistic regression model, an SVM model, and a Complement Naive Bayes model. 
    - Finally, we will evaluate their performance on the test set using the confusion matrix and F1 score.We also measure computational efficiency by recording the model training time.
4. **Ranked BERtopics by mean VADER sentiment**
We applied BERTopic to the full r/polticaldiscussion dataset (25%) after standard preprocessing (text, cleaning, stopword removal, and removal of extremely short comments). BERTopic was implemented using embeddings and clustering methods producing a set of coherent topical groupings. For each topic, we computed the mean VADER compound sentiment score across all comments assigned to that topic. This metric allowed us to assess the mean emotional valence associated with specific political themes. Manual review was used to check topic coherence, ensuring that aggregated sentiment scores reliably reflected the underlying discourse.  
5. **SD of VADER sentiment shows emotional disagreement and polarization within a topic** 
Use the same BERTopic clusters we calculated the standard deviation of VADER sentiment scores within each topic to measure emotional disagreement. While mean sentiment captures the general emotional tone, standard deviation reveals how divided users are in their responses. A high standard deviation indicates the presence of both strongly negative and strongly positive sentiment within the same topic. These metrics serve as a proxy for emotional polarization and allow us to identify issues that provoke disagreement rather than uniform reactions. 
6. **Determining high risk of misinformation in comments by combining Narrative score and Negative sentiment**
To isolate comments most likely to contain political misinformation, we used an LLM-based classifier that evaluated the factual reliability and rhetorical structure of comments. We then combined these predictions with two additional measures: (1) a narrative structure score that captures features such as causal framing, certainty, and storytelling coherence, and (2) VADER sentiment, focusing on highly negative polarity (VADER <-0.5). Comments were classified as “high-risk” only if they met the LLMs misinformation criteria and fell into the top range of narrative scores and negative sentiment. This multi-dimensional approach allowed us to detect not just false statements but persuasive misinformation style narratives. 

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