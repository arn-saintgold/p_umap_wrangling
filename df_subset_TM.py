import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sentence_transformers import SentenceTransformer
from umap import UMAP
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from encoder import build_model
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
# Import labeled subset of ~500 documents
df = pd.read_csv('df_subset.csv')[[ 'Domain','created_utc',
       'title', 'expanded_url',
       'Score', 'Rating', 'ArticleTitle', 'ArticleDate', 'ArticleText',
       'Keywords', 'Error', 'labels']]

# Import standard SBERT
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

standard_model = BERTopic()

umap_model = UMAP(n_components=5, n_neighbors=15, metric='cosine')
empty_reduction_model = BaseDimensionalityReduction()


hdb_plus = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', gen_min_span_tree=True))

ArticleEmbeddings = sentence_model.encode(df.ArticleText)
TitleEmbeddings = sentence_model.encode(df.ArticleText)

combo_embeddings = np.concat([ArticleEmbeddings, TitleEmbeddings], axis=1)
df['timestamp'] = pd.to_datetime(df['created_utc'], utc=True)
min_timestamp = df['timestamp'].min()
df['days_since_earliest'] = (df['timestamp'] - min_timestamp).dt.days
days = np.array(df['days_since_earliest'])
days = days.reshape(len(days),1)
time_array = np.sqrt(np.sqrt(days))

combo_time_embeddings = np.concat([combo_embeddings,time_array], axis=1)




def TM_diagnosis(model, docs, embeddings=None, umap_model= None, hdbscan_model = None, df = df, png_name='TMDiag', optimize_clust=False, leaf_only=False, pumap=False):
    
    if png_name:
        print(f"DIAGNOSIS FOR {png_name.upper()}")

    if embeddings is None:
        embeddings = sentence_model.encode(docs)

    def get_keys_with_max_value(d):
        if not d:
            return []
        max_val = max(d.values())
        return [k for k, v in d.items() if v == max_val]
    

    params = {}
    if optimize_clust:
        for n_components in [20,50,100, 200]:
            for n_neighbors in [5,15,50]:

                umap = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
                reduced_embeddings = umap.fit_transform(embeddings)
                for method in ['leaf','eom']:
                    if leaf_only and method=='eom':
                        continue
                    print(f"OPTIMIZING {(n_components, n_neighbors, method) =}")
                    for min_clust in range(2,21):
                        clustering = HDBSCAN(min_cluster_size = min_clust, gen_min_span_tree=True, cluster_selection_method=method )
                        clustering.fit(reduced_embeddings)
                        validity = clustering.relative_validity_
                        params[(n_components, n_neighbors, min_clust, method)] = validity
        n_components, n_neighbors, min_clust, method = get_keys_with_max_value(params)[0]
        print(f"CHOSEN PARAMS: {(n_components, n_neighbors, min_clust, method) =}")
        if pumap:

            # P UMAP INITIALIZATION

            print('INITIALIZE PARAMETRIC UMAP 🧣🧣🧣')

            dims = combo_time_embeddings[0].shape
            dimension=20
            batch_size = len(embeddings)//4+1

            encoder = build_model(
                (dims),
                head_size=2,
                num_heads=31,
                ff_dim=610,
                num_transformer_blocks=2,
                mlp_units=[928,569,564],
                mlp_dropout=0.0010533336845051193,
                dropout=0.055311304614280965,
                n_classes=n_components
                )


            keras_fit_kwargs = {"callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    min_delta=10**-2,
                    patience=10,
                    verbose=0,
                )
            ]}

            umap_args = {'n_neighbors': n_neighbors,
                        'verbose': True, 
                        'n_epochs':20,
                        'n_components': n_components,
                        'metric': 'cosine',
                        "encoder": encoder,
                        "dims": dims,
                        "batch_size": batch_size,
                        "keras_fit_kwargs":keras_fit_kwargs
                        }

            umap_model = ParametricUMAP(**umap_args)

            umap_model.save('df_subset_pumap')

        else:
            umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine')
        reduced_embeddings = umap.fit_transform(embeddings)
        hdbscan_model = HDBSCAN(min_cluster_size = min_clust,gen_min_span_tree=True, cluster_selection_method=method )
        model = BERTopic(embedding_model=sentence_model,
                         umap_model = umap_model,
                         hdbscan_model = hdbscan_model
                         )

    model.fit_transform(docs, embeddings)

    doc_df = model.get_document_info(docs)
    topic_df = model.get_topic_info()
    
    df = df.merge( doc_df[['Document', 'Topic', 'Representation']], left_on = 'ArticleText', right_on = 'Document')
    df['isNoise'] = df['Topic']==-1
    noise_pct = str(float(round(np.mean(df['isNoise'])*100, 2)))+'%'
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    #plt.figure(figsize=(10, 4))
    sns.histplot(data=topic_df.query('Topic != -1'), x='Count', fill=True, ax=axs[0])
    axs[0].set_title('Noise%: '+noise_pct+'. Topic Sizes Histogram: ')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Numerosity')
    axs[0].grid(True)
    #plt.show()
    
    # ---- 2. Donut plots for Top grouped by Rel == -1 and Rel == 0 ----
    
    def plot_donut(data, rel_value, ax):
        subset = data[data['Rating'] == rel_value]
        counts = subset['isNoise'].value_counts()
        labels = counts.index
        sizes = counts.values
        total = sizes.sum()
        percentages = [f"{v / total:.1%}" for v in sizes]
    
        wedges, texts = ax.pie(
            sizes,
            labels=[f"{l} ({p})" for l, p in zip(labels, percentages)],
            startangle=90,
            wedgeprops=dict(width=0.3),
            counterclock=False
        )
        ax.set_title(f"Noise Distribution (Reliability = {rel_value})")
    
    # Create side-by-side donut plots
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_donut(df, rel_value='T', ax=axs[1])
    plot_donut(df, rel_value='N', ax=axs[2])
    #plt.show()
    plt.tight_layout()
    plt.savefig(png_name+".png", dpi=300)


#TM_diagnosis(standard_model, df.ArticleText, png_name='Std_Diag')

#TM_diagnosis(hdb_plus, df.ArticleText, png_name='LeafHDBSCAN_Diag')

#TM_diagnosis(None, df.ArticleText, png_name='OptStdLeaf_Diag', optimize_clust=True, leaf_only=True)

#TM_diagnosis(None, df.ArticleText, embeddings = combo_embeddings, png_name='OptStdComboLeaf_Diag', optimize_clust=True)

#TM_diagnosis(None, df.ArticleText, embeddings = combo_time_embeddings, png_name='OptStdComboTime_Diag', optimize_clust=True)

#TM_diagnosis(None, df.ArticleText, embeddings = combo_time_embeddings, png_name='OptStdComboTimeLeaf_Diag', optimize_clust=True, leaf_only=True)

TM_diagnosis(None, df.ArticleText, embeddings = combo_time_embeddings, png_name='PumapOptStdComboTimeLeaf_Diag', optimize_clust=True, pumap=True)
