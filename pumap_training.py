import os
import datetime
import ast
import io
import pickle
import pandas as pd
import numpy as np
from minio import Minio
from minio_helper import read_file_minio
from minio_download import GetMinioArticles
from encoder import build_model
from dotenv import load_dotenv
from umap.parametric_umap import ParametricUMAP
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import trustworthiness
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from save_encoded_articles import Embedder
import torch
import tensorflow as tf
import keras
from umap.parametric_umap import load_ParametricUMAP

now = datetime.datetime.now

print('READ MISTRAL EMBEDDINGS FROM MINIO')

# get BytesIO stream object
client = Minio(
    os.getenv("minio_endpoint"),
    access_key = os.getenv("minio_access_key"),
    secret_key = os.getenv("minio_secret_key"),
    secure = False
        )
    
bucket_name = os.getenv("bucket_name")

io_stream_object = read_file_minio(client=client, bucket_name=bucket_name, obj_name='mistral/embeddings/mistral_embeddings.pkl' ,obj_type='pkl')
# Create a BytesIO stream and pickle an object into it
buffer = io.BytesIO()

# Unpickle from the BytesIO stream
mistral_embeddings = pickle.load(io_stream_object)

#mistral_embeddings = mistral_embeddings.numpy()
mistral_embeddings = tf.convert_to_tensor(mistral_embeddings)

print(mistral_embeddings.shape)
print(type(mistral_embeddings))
dims = mistral_embeddings[0].shape
print('INITIALIZE PARAMETRIC UMAP 🧣🧣🧣')


dimension=20
batch_size = 4#1024*4

'''
print(f"{mistral_embeddings.shape=}")

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
'''

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
             'n_epochs':20,
             'n_components': dimension,
             'metric': 'cosine',
             #"encoder": encoder,
             "dims": dims,
             "batch_size": batch_size,
             "keras_fit_kwargs":keras_fit_kwargs
            }

# Initialize and fit the model
pumap = ParametricUMAP(**umap_args)

print('LEARNING DATA 🪡')
# Fit and transform the data

embeddings = pumap.fit_transform(mistral_embeddings)
#pumap.save('')
pumap.save('transformer_umap')

reducer = load_ParametricUMAP('transformer_umap')
print('Model Loaded Successfully')
quit()
