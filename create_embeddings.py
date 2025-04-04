import datetime
import ast
import pandas as pd
import numpy as np
from encoder import build_model
from dotenv import load_dotenv
from umap.parametric_umap import ParametricUMAP
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import trustworthiness
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from save_encoded_articles import Embedder
from minio_download import GetMinioArticles
import tensorflow as tf
import keras
import pickle

now = datetime.datetime.now

print('READING DATA 📚')
df = pd.read_csv('scraped_news.csv')
df = df[:len(df)//3]

texts = df.ArticleTitle + df.ArticleText
texts = [str(x) for x in texts]
#articles = [str(x) for x in df.ArticleText]
#titles = [str(x) for x in df.ArticleTitle]
keywords = [', '.join(ast.literal_eval(k)) for k in df.Keywords]

embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
embedder = Embedder(embedding_model = embedding_model_name)
print('EMBEDDING DATA 🧩')
embedder.embed_n_concat(texts, keywords)

embeddings = embedder.embeddings
print(f"{len(embeddings)=}")

dims = embeddings[0].shape
print(f'{dims=}')

with open('embeddings.pkl','wb') as handle:
    pickle.dump(embeddings, handle)