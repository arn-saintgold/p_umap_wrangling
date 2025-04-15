import os
import io
import ast
from minio import Minio
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sentence_transformers import SentenceTransformer
from minio_helper import read_file_minio
import pickle 

# --- CONFIGURATION ---
original_dim = 4096         # Original vector size
reduced_dim = 20#300        # Target reduced dimension
batch_size = 1000           # Size of each data chunk
data_file = "temp/mistral_n_key_embeddings.npy"  # Large .npy file containing embeddings

os.makedirs('temp', exist_ok=True)
os.makedirs('processed_embeddings', exist_ok=True)

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

# --- Step 0: Filter out duplicates ---
df = read_file_minio(client=client, bucket_name=bucket_name, obj_name='mistral/data/scraped_news.csv' ,obj_type='csv')
# compute keywords embedding
df = df.drop_duplicates(subset='ArticleText', keep=False)
keywords = [', '.join(ast.literal_eval(k)) for k in df.Keywords]

embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(embedding_model_name, device='cuda')
print('EMBEDDING KEYWORDS 🧩')
key_embeddings = sentence_model.encode(keywords, show_progress_bar=False)
print('KEYWORDS EMBEDDED')

# Step 1: Identify duplicates in column 'A'
duplicates_mask = df.duplicated(subset='ArticleText', keep=False)

# Step 2: Use the inverse of the mask to filter out corresponding elements in arr
mistral_embeddings = pickle.load(io_stream_object)
mistral_embeddings = mistral_embeddings[~duplicates_mask.to_numpy()]

mistral_embeddings = np.concat([mistral_embeddings, key_embeddings], axis=1)

np.save('temp/mistral_n_key_embeddings.npy',mistral_embeddings)

del mistral_embeddings


# --- Load data in chunks ---
def data_generator(filename, batch_size):
    with open(filename, 'rb') as f:
        # Load shape info without reading entire array
        shape = np.load(f, allow_pickle=True).shape
        total = shape[0]
        
        # Use memory mapping to read data lazily
        mmap = np.load(filename, mmap_mode='r')
        
        for i in range(0, total, batch_size):
            yield mmap[i:i + batch_size]

# --- Step 1: Fit IPCA incrementally ---
ipca = IncrementalPCA(n_components=reduced_dim, batch_size=batch_size)

print("Fitting IPCA...")
for chunk in data_generator(data_file, batch_size):
    ipca.partial_fit(chunk)

# --- Step 2: Transform full dataset in chunks ---
print("Transforming data...")
reduced_data = []

for chunk in data_generator(data_file, batch_size):
    reduced_chunk = ipca.transform(chunk)
    reduced_data.append(reduced_chunk)

# --- Combine and save result ---
reduced_array = np.vstack(reduced_data)
np.save("processed_embeddings/reduced_embeddings.npy", reduced_array)

print(f"Reduced embeddings saved to 'reduced_embeddings.npy'. Shape: {reduced_array.shape}")
