import pandas as pd
import phenograph2 #
import phenograph
import sys
sys.path.append(r'/exports/reum/tdmaarseveen/RA_Clustering/src/1_emr_scripts')
import importlib as imp
import Visualization as func
from phenograph2.cluster import cluster as cluster2
import umap 
from sklearn.manifold import TSNE
from sklearn.utils import resample
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt


## -------------------------------------------DEFINE GLOBALS----------------------------------------
EMBEDDING_SPACE = 'MMAE_embedding' # maui_embedding_3
PROJECTION_DATA = 'MMAE_clustering_270' 


## -------------------------------------------CREATE LIST W/ CATEGORICAL VARIABLES----------------------------------------

l_cat = ['RF', 'aCCP', 'Sex']
df_meta = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/4_processed/DF_Mannequin_NEW_Engineered.csv', sep='|')
l_keep = [x for x in list(df_meta.columns) if x not in ['patnr', 'FirstConsult', 'pseudoId']]
l_cat.extend(l_keep)

## ---------------------------------------------------IMPORT RELEVANT DATA------------------------------------------------

# Latent feature space (a.k.a. Factors from MOFA)
Z = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/results/embedding/%s.csv' % EMBEDDING_SPACE) 

Z = Z[Z.columns[4:]]

# Acquire clusters (Loading final dataset
metadata = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/%s.csv' % PROJECTION_DATA, index_col=0)


## ---------------------------------------------------PERFORM BOOTSTRAPPING------------------------------------------------

print('yea')

# bootstrap configurations
k = 270 
l_ari = []
n_iterations = 1500 # 1000 #1500
l_col = [] 
SEED = 20221212
ENFORCE_NUMBER_CLUSTERS = 4

ks = np.arange(1, n_iterations+1, 1)
n = len(ks)


df_booted = pd.DataFrame(np.empty((len(Z),0,)))
 
for i in range(n_iterations):
    X_bs = Z.loc[np.random.choice(Z.index, size=len(Z), replace=True)].copy()
    l_samples = list(X_bs.index)
    
    # sort by index - to reduce amount of noise (prevent same patients being assigned to different clusters)
    #X_bs = X_bs.sort_index()
    
    X = np.array(X_bs)
    
    communities, graph, Q = cluster2(X_bs, k=k, seed=SEED, primary_metric='minkowski', n_jobs=1) # minkowski
    
    if ENFORCE_NUMBER_CLUSTERS in list(set(communities)):
        print('SKIPPED (Reason: More clusters than expected!)')
        continue  # Skip to next iteration
    
    cluster_name = 'PhenoGraph_iter_' + str(i) 
    l_col.append(cluster_name)
    X_bs[cluster_name] = pd.Categorical(communities)
    
    

    df_booted = df_booted.merge(X_bs[cluster_name], left_index=True, right_index=True, how='left')
    if len(df_booted[df_booted.index.duplicated(keep=False)]) > 1 :
        sub_df = df_booted[df_booted.index.duplicated(keep=False)]
        l_unique_ix = list(sub_df.index.unique())

        for pat in l_unique_ix:
            if len(sub_df.loc[pat][cluster_name].unique())>1:
                print('[Iteration %s] Warning: Patient %s is assigned to multiple clusters: ' % (i, pat),  sub_df.loc[pat][cluster_name])
                #print(eql)
    df_booted = df_booted[~df_booted.index.duplicated(keep='first')] # keep first
    
    
        
    

metadata = pd.merge(metadata,df_booted, left_index=True, right_index=True) 

print('No duplicate entries')

## ------------------------------------ CALCULATE SIMILARITY MATRIX---------------------------------------


def hamming_nan(u, v):
    """
    Nans should be replaced with negative values.
    
    During clustering comparison, we ignore the patients that are missing 
    in either one of the clusterings. Whereby the nan's in either vector
    inform the eventual indexing.
    """
    l_miss = list((u <0).nonzero()[0])
    l_miss.extend(list((v <0).nonzero()[0]))
    l_pres =  list(set([i for i in range(len(u))]) - set(l_miss))
    if len(l_pres)==0:
        return 1 # max distance
    else :
        return sum(np.array(u)[l_pres]!=np.array(v)[l_pres])/len(l_pres)

n = len(metadata)
ks = np.arange(1, n+1, 1)
Similarity_indices = pd.DataFrame(np.zeros((n,n)), index = ks, columns = ks)

Similarity_indices.index.name = 'k1'
Similarity_indices.columns.name = 'k2'

df_resamp = metadata.sort_values(by='PhenoGraph_clusters').copy()

l_cols = [col for col in df_resamp.columns if 'PhenoGraph_iter_' in col]

col_1 = 'PhenoGraph_iter_' + str(min([int(i.split('PhenoGraph_iter_')[1]) for i in l_cols]))
col_2 = 'PhenoGraph_iter_' + str(max([int(i.split('PhenoGraph_iter_')[1]) for i in l_cols]))

# fill na temporarily with -1 -> we remove those later
df_resamp.loc[:, col_1:col_2:1] = df_resamp.loc[:, col_1:col_2:1].astype(float).fillna(-1)
df_resamp = df_resamp.reset_index()

cluster_labels = df_resamp["PhenoGraph_clusters"].astype(str)
cluster_pal = sns.color_palette() 
cluster_lut = dict(zip(map(str, cluster_labels.unique()), cluster_pal))

cluster_colors = pd.Series(cluster_labels).map(cluster_lut)



Similarity_indices = squareform(pdist(np.array(df_resamp.loc[:, col_1:col_2:1]), hamming_nan))

## ------------------------------------ CALCULATE CO-CLUSTER PROBABILITY--------------------------------------
from scipy.spatial import distance

n = len(metadata)

for cluster in metadata["PhenoGraph_clusters"].unique():
    print(cluster)
    l_indices = df_resamp[df_resamp["PhenoGraph_clusters"]==cluster].index.values.tolist()
    l_proba_cluster = []
    for ix, row in df_resamp.iterrows():
        # how often is this patient assigned to the correct cluster
        l_proba = [1- Similarity_indices[ix, j] for j in l_indices if j != ix]
        l_proba_cluster.append(np.mean(l_proba))
    df_resamp['PROBA_CLUSTER%s'% (cluster)] = l_proba_cluster

## ------------------------------------ CALCULATE PROBABILITY--------------------------------------

# Add cluster probability for each cluster that you found
for cluster in metadata['PhenoGraph_clusters'].unique():
    d_proba = dict(zip(df_resamp.pseudoId, df_resamp['PROBA_CLUSTER%s'% (cluster)])) 
    metadata['PROBA_CLUSTER%s' % (cluster)] = metadata['pseudoId'].apply(lambda x: d_proba[x] if x in d_proba.keys() else None)

# Color the samples where MTX is missing grey (because we don't use these for the downstream analysis)
metadata.to_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/%s_Stability.csv' % PROJECTION_DATA)

## ------------------------------------ CREATE 2D Embedding--------------------------------------
l_final_col = [ i for i in metadata.columns if 'PhenoGraph_iter_' not in i ]#[2:]

# Factors
fit1 = TSNE(random_state=SEED, n_components=2, perplexity=30, metric='minkowski').fit_transform(Z.values)
fit2 = umap.UMAP(random_state=SEED+96812,metric='minkowski', n_neighbors=7).fit(Z.values).embedding_

func.visualize_umap_bokeh_na_2(fit1, metadata[l_final_col], list(l_final_col), l_binary=l_cat, patient_id='sample', cluster_id='PhenoGraph_clusters', title='%s_STABILITY_TSNE'  % PROJECTION_DATA)
func.visualize_umap_bokeh_na_2(fit2, metadata[l_final_col], list(l_final_col), l_binary=l_cat, patient_id='sample', cluster_id='PhenoGraph_clusters', title='%s_STABILITY_UMAP' % PROJECTION_DATA)


## Plot heatmap 
g = sns.clustermap(1-Similarity_indices,
                    col_colors=cluster_colors.to_numpy(), 
                  # Turn off the clustering
                  #row_cluster=True, col_cluster=False,

                  # Add colored class labels # col_colors=cluster_colors,
                   cmap="CMRmap_r",  # vmax=5 ,  #vmin=-2, # vmax=5 ,  

                  # Make the plot look better when many rows/cols
                  linewidths=0, xticklabels=False, yticklabels=True) # , cmap="PiYG")

for label in cluster_labels.unique():
    g.ax_col_dendrogram.bar(0, 0, color=cluster_lut[label],
                            label=label, linewidth=0)
g.ax_col_dendrogram.legend(loc="center", ncol=6)

g.cax.set_position([-.15, .2, .03, .45]) 

plt.savefig("/exports/reum/tdmaarseveen/RA_Clustering/figures/3_clustering/heatmap_unsorted.png") 
