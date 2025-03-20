import pandas as pd
import numpy as np
from encoder import build_model
from dotenv import load_dotenv
from parametric_umap import ParametricUMAP
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import trustworthiness
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

load_dotenv()



# Create sample data
#n_samples = 1000
#X, color = make_swiss_roll(n_samples=n_samples, random_state=42)
print('READING DATA 📚')
data = pd.read_csv('scraped_news.csv').ArticleText
print('EMBEDDING DATA 🧩')
embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
embedding_model=SentenceTransformer(embedding_model_name)
embedded_docs = embedding_model.encode(data, show_progress_bar=True)
print('SPLITTING DATA ✂')
X, X_new, = train_test_split(embedded_docs, test_size=0.33, random_state=42)

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
dims = X[0].shape
dimension=20
device='cuda:0'
batch_size = 1024*8
umap_args = {'n_neighbors': 15,
             'device' : device
             'n_components': dimension,
             'metric': 'cosine',
             "encoder": encoder,
             "dims": dims,
             "batch_size": batch_size,
             "keras_fit_kwargs":{"verbose":1}}


print(f"{X.shape=}")

print('INITIALIZE PARAMETRIC UMAP 🧣🧣🧣')
# Initialize and fit the model
pumap = ParametricUMAP(**umap_args)
pumap = ParametricUMAP(
    device='cuda:0',
    n_components=20,
    hidden_dim=128,
    n_layers=3,
    n_epochs=10
)

print('LEARNING DATA 🪡')
# Fit and transform the data
embeddings = pumap.fit_transform(X)#, low_memory=True)

#print('MEASURING PERFORMANCE ON TRAIN DATA 📏')
# measure trustworthyness
#trust = trustworthiness(X, embeddings, n_neighbors=15, metric='euclidean')
#cont = trustworthiness(embeddings, X, n_neighbors=15, metric='euclidean')
#print(f'trustworthiness: {round(trust*100, 2)}%')
#print(f'continuity: {round(cont*100, 2)}%')

print('MEASURING PERFORMANCE ON TEST DATA 📏')
# Transform new data
new_embeddings = pumap.transform(X_new)

trust = trustworthiness(X_new, new_embeddings, n_neighbors=5, metric='euclidean')
#cont = trustworthiness(new_embeddings, X_new, n_neighbors=15, metric='euclidean')
print(f'trustworthiness: {round(trust*100, 2)}%')
print(f'continuity: {round(cont*100, 2)}%')

