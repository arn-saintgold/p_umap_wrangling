import ast
import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
from minio import Minio
from minio_download import GetMinioArticles
from save_encoded_articles import Embedder
import minio_helper
from umap.parametric_umap import ParametricUMAP
from umap.parametric_umap import load_ParametricUMAP
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.model_selection import train_test_split


# Fit BERTopic without actually performing any dimensionality reduction

def main():
    load_dotenv()
    
    print([os.getenv('bucket_name'),os.getenv('minio_endpoint')])
    
    import pandas as pd
    print('DOWNLOAD DATA FROM MINIO')
    def my_query(df: pd.DataFrame) -> pd.DataFrame:
        df = df.query("Error.isnull() and not ArticleText.isnull()").drop_duplicates(subset=['ArticleText'], keep=False)
        
        return df[['Domain','expanded_url', 'created_utc', 'Rating', 'ArticleTitle','ArticleText', 'Keywords']]
        
    df = GetMinioArticles().get_articles_from_minio(my_query=my_query, no_save=False, to_minio = False)
    print(df.columns)
    print(f'{len(df)=}')
    print('TEST FINISHED')

    texts = df.ArticleTitle + df.ArticleText
    texts = [str(x) for x in texts]
    
    #start federetad learning
    
    precomputed_embeddings = False
    if precomputed_embeddings:
        with open('../p_umap_wrangling/embeddings.pkl','rb') as handle:
            embeddings = pickle.load(handle)
    # EMBED TEXTS
    else:
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
    
    reducer = load_ParametricUMAP('../p_umap_wrangling/transformer_umap')

    # create empty umap
    empty_dimensionality_model = BaseDimensionalityReduction()
    
    n_splits = len(embeddings)//25_000 +1
    
    # SPLIT EMBEDDINGS

    # Split the indices
    indices = np.array_split(np.arange(len(embeddings)), n_splits)

    models = []

    assert len(embeddings) == len(texts), "Array and list must have the same length!"
    
    # Iterate through splits
    for i, idx in enumerate(indices):
        print(f'FITTING MODEL {i}')
        print(embeddings.shape)

        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', gen_min_span_tree=True)
        vectorizer_model = CountVectorizer(min_df=3,stop_words='english', ngram_range=(1,2))
        ctfidf_model = ClassTfidfTransformer()
        
        print(f'REDUCE EMBEDDINGS')

        emb_split = reducer.transform(embeddings[idx])  # Select rows from the NumPy array
        text_split = [texts[j] for j in idx]  # Select rows from the text list

        print(f"CREATE TOPIC MODEL")
        # Create topic models
        topic_model = BERTopic(umap_model=empty_dimensionality_model,
                               min_topic_size=5,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model,
                               ctfidf_model=ctfidf_model)
        topic_model.fit(text_split, emb_split)
        models.append(topic_model) 
    
    print('MERGING MODELS')
    merged_model = BERTopic.merge_models(models)
    models = [merged_model]
    
    topic_df = merged_model.get_document_info(df.ArticleText)
    N_topics = len(merged_model.get_topic_info)
    print(f"{N_topics} FOUND")
    
    noise_idx = np.where(topic_df.Topic == -1)
    noise_texts = topic_df.iloc[noise_idx].ArticleText
    noise_embeddings = embeddings[noise_idx]

    # SPLIT EMBEDDINGS
    n_splits = len(noise_embeddings)//25_000 +1
    
    indices = np.array_split(np.arange(len(noise_embeddings)), n_splits)

    for i, idx in enumerate(indices):

        print(f'FITTING NOISE MODEL {i}')
        print(f'{noise_embeddings.shape=}')        
        
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', gen_min_span_tree=True)
        vectorizer_model = CountVectorizer(min_df=3,stop_words='english', ngram_range=(1,2))
        ctfidf_model = ClassTfidfTransformer()
        
        print(f'REDUCE NOISE EMBEDDINGS')
        
        emb_split = reducer.transform(noise_embeddings[idx])  # Select rows from the NumPy array
        text_split = [noise_texts[j] for j in idx]  # Select rows from the text list

        print(f"CREATE NOISE TOPIC MODEL")
        # Create topic models
        topic_model = BERTopic(umap_model=empty_dimensionality_model,
                               min_topic_size=5,
                               hdbscan_model=hdbscan_model,
                               vectorizer_model=vectorizer_model,
                               ctfidf_model=ctfidf_model)
        topic_model.fit(text_split, emb_split)
        models.append(topic_model) 

    merged_model = BERTopic.merge_models(models)    
    
    
    print('SAVING MODEL')
    os.makedirs('topic_models', exist_ok = True)
    merged_model.save(os.path.join('topic_models','merged.topic.model'), save_embedding_model=True)

    print(f"{len(merged_model.get_topic_info()) - N_topics} NEW TOPICS FOUND")

    print('END')
if __name__ == '__main__':
    main()
