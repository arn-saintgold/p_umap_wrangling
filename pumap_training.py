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

#def my_query(df: pd.DataFrame) -> pd.DataFrame:
#    df = df.query("Error.isnull()")
#    
#    return df[['Domain', 'created_utc', 'Rating', 'ArticleTitle','ArticleText', 'Keywords']]
#
#df = GetMinioArticles().get_articles_from_minio(my_query=my_query)

print('READING DATA 📚')
df = pd.read_csv('scraped_news.csv')
#df = df[:len(df)//2]

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

print('INITIALIZE PARAMETRIC UMAP 🧣🧣🧣')

dimension=20
batch_size = 1024*8
print(f"{embeddings.shape=}")

encoder = build_model(
        (dims),
        head_size=2,
        num_heads=31,
        ff_dim=610,
        num_transformer_blocks=2,
        mlp_units=[928,569,564],
        mlp_dropout=0.0010533336845051193,
        dropout=0.055311304614280965,
        n_classes=dimension
        )
keras_fit_kwargs = {"callbacks": [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=10**-2,
        patience=10,
        verbose=0,
    )
]}

umap_args = {'n_neighbors': 15,
             'verbose': True, 
             'n_epochs':3,
             #'device' : device,
             'n_components': dimension,
             'metric': 'cosine',
             "encoder": encoder,
             "dims": dims,
             "batch_size": batch_size,
             "keras_fit_kwargs":{'verbose':0}#keras_fit_kwargs
            }

# Initialize and fit the model
pumap = ParametricUMAP(**umap_args)

print('LEARNING DATA 🪡')
# Fit and transform the data
embeddings = pumap.fit_transform(embeddings)#, low_memory = True)
pumap.save('')
