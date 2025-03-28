import ast
import pickle
import numpy as np
import pandas as pd
from save_encoded_articles import Embedder

print('READING DATA 📚')
df = pd.read_csv('scraped_news.csv')
articles = [str(x) for x in df.ArticleText]
titles = [str(x) for x in df.ArticleTitle]
keywords = [', '.join(ast.literal_eval(k)) for k in df.Keywords]

embedder = Embedder

