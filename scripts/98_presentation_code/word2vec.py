# train_word2vec.py
import pandas as pd
import re
from tqdm import tqdm
from gensim.models import Word2Vec



df = pd.read_csv(".././data/reddit_sample.csv")


texts = df["body"].fillna("").astype(str).tolist()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

sentences = [clean_text(t).split() for t in tqdm(texts, desc="Cleaning texts")]


model = Word2Vec(
    sentences=sentences,
    vector_size=100,     
    window=5,            
    min_count=3,        
    workers=4,           
    sg=1,               
    epochs=5
)


model.save(".././data/reddit_word2vec.model")
model.wv.save_word2vec_format(".././data/reddit_word2vec.txt")
