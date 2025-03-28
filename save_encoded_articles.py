import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from minio import Minio
import minio_helper


class Embedder:

    def __init__(self, embedding_model:str):
        self.embedding_model = embedding_model
        self.client = Minio(
            os.getenv("minio_endpoint"),
            access_key = os.getenv("minio_access_key"),
            secret_key = os.getenv("minio_secret_key"),
            secure = False
        )
    
        self.bucket_name = os.getenv("bucket_name")

    def save_pickle_to_minio(self, data, output_path, subreddit_name = 'news'):
    
        self.subreddit_name = subreddit_name

        minio_helper.save_file_minio(self.client, self.bucket_name, output_path, obj_to_upload=data, obj_type='pickle')

    
    def save_embeddings_to_minio(self, output_path, subreddit_name = 'news'):

        self.save_pickle_to_minio(data=self.embeddings, output_path=output_path, subreddit_name=subreddit_name)

    
    def get_embedding_model(self):
        import torch

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()
        
        #Additional Info when using cuda
        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        print('DOWNLOADING EMBEDDER 🤖')
        self.sentence_model = SentenceTransformer(self.embedding_model, device=device)
    
    def embed(self, data):
        self.get_embedding_model()
        print("EMBEDDING THE EMBEDDABLE 🛏 … 🚶")
        self.embeddings = self.sentence_model.encode(data, show_progress_bar=False)
        print("EMBEDDABLE EMBEDDED 🛌")

    def embed_n_concat(self, *args):
        self.get_embedding_model()
        features = args
        if len(features)<2:
            self.embed(data=args[0])
            
        else:
            print('EMBEDDING FEATURE 1 🛏 … 🚶')
            self.embeddings = self.sentence_model.encode(features[0], show_progress_bar=False)
            
            for i,feature in enumerate(features[1:]):
                print(f'EMBEDDING FEATURE {i+2} 🛏 … 🚶')
                embeddings = self.sentence_model.encode(feature, show_progress_bar=False)
                self.embeddings = np.concatenate([self.embeddings,embeddings], axis=1)
            print("EMBEDDABLE EMBEDDED 🛌")
        
    def get_embeddings(self):
        if self.embeddings is None:
            raise BaseException("Trying to access embeddings before initialization")
        return self.embeddings

    def pickle_embeddings(self, pickle_path='', output_path='', to_minio=True):
        if self.embeddings is None:
            raise BaseException("Trying to access embeddings before initialization")
        if not to_minio:
            with open(pickle_path, 'wb') as pkl:
                pickle.dump(self.embeddings, pkl)
        if to_minio:
            self.save_pickle_to_minio( self.embeddings, output_path)
        print("PICKLE SAVED 🥒")

    def embed_and_save(self, data, pickle_path='embeddings.pkl', output_path=os.path.join('urls_and_articles', 'embeddings'+'_news'), to_minio=True):
        self.embed(data)
        self.pickle_embeddings(pickle_path, output_path=output_path, to_minio=to_minio)

    def get_embeddings_from_minio(self, minio_path=os.path.join('urls_and_articles','embeddings'+'_news')):
        
        embeddings = minio_helper.read_file_minio(client = self.client,
                                                bucket_name = self.bucket_name,
                                                obj_name = minio_path,
                                                obj_type = 'pickle')
        return embeddings
    
def main():
    from minio_download import GetMinioArticles

    def my_query(df: pd.DataFrame) -> pd.DataFrame:
        df = df.query("Error.isnull()")
        
        return df[['Domain', 'created_utc', 'Rating', 'ArticleTitle','ArticleText', 'Keywords']]
        
    df = GetMinioArticles().get_articles_from_minio(my_query=my_query)
    texts = df['ArticleText'].astype(str).to_list()

    print(f"{len(texts)=}")
    
    embedding_model = 'paraphrase-multilingual-mpnet-base-v2'
    embedder = Embedder(embedding_model = embedding_model)
    embedder.embed_and_save(texts)
    embeddings = embedder.get_embeddings_from_minio()
    print(f"{len(embeddings)=}")

if __name__=='__main__':
    main()
