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
from umap.parametric_umap import load_ParametricUMAP

now = datetime.datetime.now

with open('embeddings.pkl','rb') as handle:
    embeddings = pickle.load(handle)




print('INITIALIZE PARAMETRIC UMAP 🧣🧣🧣')

# Initialize and fit the model
pumap = ParametricUMAP()#**umap_args)

print('LEARNING DATA 🪡')
# Fit and transform the data
embeddings = pumap.fit_transform(embeddings)
#pumap.save('')
pumap.save('transformer_umap')

reducer = load_ParametricUMAP('transformer_umap')
print('Model Loaded Successfully')
quit()

dimension=20
batch_size = 1024*4
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
             'n_epochs':10,
             'n_components': dimension,
             'metric': 'cosine',
             #"encoder": encoder,
             "dims": dims,
             "batch_size": batch_size,
             "keras_fit_kwargs":keras_fit_kwargs
            }

