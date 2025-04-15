import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import matplotlib.pyplot as plt
import seaborn as sns
# Import labeled subset of ~500 documents
df = pd.read_csv('df_subset.csv')

# Import standard SBERT
#sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

standard_model = BERTopic()

hdb_plus = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='leaf', gen_min_span_tree=True)

#sdtPlus_model = BERTopic(hdbscan_model=hdb_plus)

def TM_diagnosis(model, docs, embeddings=None, df = df, png_name='TMDiag'):

    standard_model.fit_transform(docs, embeddings)

    doc_df = standard_model.get_document_info(docs)
    
    df = df.merge( doc_df[['Document', 'Topic', 'Representation']], left_on = 'ArticleText', right_on = 'Document')
    df['isNoise'] = df['Topic']==-1
    noise_pct = str(float(round(np.mean(df['isNoise'])*100, 2)))+'%'
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    #plt.figure(figsize=(10, 4))
    sns.kdeplot(data=topic_df.query('Topic != -1'), x='Count', fill=True, ax=axs[0])
    axs[0].set_title('Noise%: '+noise_pct+'. Topic Sizes Density: ')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Density')
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

TM_diagnosis(standard_model, df.ArticleText, png_name='Std_Diag')
