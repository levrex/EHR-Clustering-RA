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
# EMBEDDING_SPACE = 'MMAE_embedding_CAT' # ?
PROJECTION_DATA = 'MMAE_clustering_240_CATEGORIC' # all_data_MAUI_clustering_240_NEW

## -------------------------------------------CREATE LIST W/ CATEGORICAL VARIABLES----------------------------------------

l_cat = ['RF', 'aCCP', 'Sex']

df_meta = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/4_processed/DF_Mannequin_NEW_Engineered.csv', sep='|')
l_keep = [x for x in list(df_meta.columns) if x not in ['patnr', 'FirstConsult', 'pseudoId']]
l_cat.extend(l_keep)

## ---------------------------------------------------IMPORT RELEVANT DATA------------------------------------------------

# Acquire clusters (Loading final dataset
metadata = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/%s.csv' % PROJECTION_DATA, index_col=0)

# Latent feature space (a.k.a. Factors from MOFA)

Z = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/5_clustering/df_categoric_wAge.csv', sep=',') # _wAge
Z = Z[Z.columns[1:]]

metadata['Age_Early'] = Z['Age_Early']
metadata['Age_Late'] = Z['Age_Late']

l_cat.extend(['Age_Late', 'Age_Early'])






## ---------------------------------------------------PERFORM BOOTSTRAPPING------------------------------------------------

print('yea')

# bootstrap predictions
k = 200 #
l_ari = []
n_iterations = 1500 
l_col = []  
SEED = 20221212 
ENFORCE_NUMBER_CLUSTERS = 4


ks = np.arange(1, n_iterations+1, 1)
n = len(ks)


df_booted = pd.DataFrame(np.empty((len(Z),0,)))
 
for i in range(n_iterations):
    X_bs = Z.loc[np.random.choice(Z.index, size=len(Z), replace=True)].copy()
    l_samples = list(X_bs.index)
    
    X = np.array(X_bs)
    
    communities, graph, Q = cluster2(X_bs, k=k, seed=SEED, primary_metric='minkowski', n_jobs=1) # minkowski
    
    if ENFORCE_NUMBER_CLUSTERS in list(set(communities)): # if more clusters than expected (remember python counts from 0)
        print('SKIPPED (Reason: More clusters than expected!)')
        continue  # Skip to next iteration

    cluster_name = 'PhenoGraph_iter_' + str(i) 
    l_col.append(cluster_name)
    X_bs[cluster_name] = pd.Categorical(communities)

    df_booted = df_booted.merge(X_bs[cluster_name], left_index=True, right_index=True, how='left')
    df_booted = df_booted[~df_booted.index.duplicated(keep='first')]

metadata = pd.merge(metadata,df_booted, left_index=True, right_index=True) 

## ------------------------------------------------EHR ------------------------------------------------

df_meta = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/offshoots/df_ehr_DAS.csv', sep='|') # WARNING: made in notebook 7
d_event_1y = dict(zip(df_meta.pseudoId, df_meta.event_1y))
d_event = dict(zip(df_meta.pseudoId, df_meta.event))
d_time = dict(zip(df_meta.pseudoId, df_meta.time))
d_mtx = dict(zip(df_meta.pseudoId, df_meta['MTX-starter']))
d_followup = dict(zip(df_meta.pseudoId, df_meta['totalFollowUp']))

metadata['pseudoId'] = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/5_clustering/df_metadata_keys.csv')['pseudoId']
metadata['event_1y'] = metadata['pseudoId'].apply(lambda x: int(d_event_1y[x]) if x in d_event_1y.keys() else None)
metadata['event'] = metadata['pseudoId'].apply(lambda x: int(d_event[x]) if x in d_event.keys() else None)
metadata['time'] = metadata['pseudoId'].apply(lambda x: d_time[x] if x in d_time.keys() else None)
metadata['MTX'] = metadata['pseudoId'].apply(lambda x: int(d_mtx[x]) if x in d_mtx.keys() else None)
metadata['totalFollowUp'] = metadata['pseudoId'].apply(lambda x: int(d_followup[x]) if x in d_followup.keys() else None)


## ------------------------------------ CALCULATE SIMILARITY MATRIX---------------------------------------


def hamming_nan(u, v):
    """
    Nans should be replaced with negative valeus
    Then they are ignored when calculating the hamming distance
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


# fill na with -1
df_resamp.loc[:, col_1:col_2:1] = df_resamp.loc[:, col_1:col_2:1].astype(float).fillna(-1)
df_resamp = df_resamp.reset_index()

cluster_labels = df_resamp["PhenoGraph_clusters"].astype(str)
cluster_pal = sns.color_palette() 
cluster_lut = dict(zip(map(str, cluster_labels.unique()), cluster_pal))

cluster_colors = pd.Series(cluster_labels).map(cluster_lut)



Similarity_indices = squareform(pdist(np.array(df_resamp.loc[:, col_1:col_2:1]), hamming_nan))

## ------------------------------------ CALCULATE PROBABILITY--------------------------------------
from scipy.spatial import distance

n = len(metadata)

for cluster in metadata["PhenoGraph_clusters"].unique():
    print(cluster)
    l_indices = df_resamp[df_resamp["PhenoGraph_clusters"]==cluster].index.values.tolist()
    l_proba_cluster = []
    for ix, row in df_resamp.iterrows():
        #print(row)
        l_proba = [1- Similarity_indices[ix, j] for j in l_indices if j != ix]
        l_proba_cluster.append(np.mean(l_proba))
    df_resamp['PROBA_CLUSTER%s'% (cluster)] = l_proba_cluster

## ------------------------------------ CALCULATE PROBABILITY--------------------------------------

# Add cluster probability for each cluster that you found
for cluster in metadata['PhenoGraph_clusters'].unique():
    d_proba = dict(zip(df_resamp.pseudoId, df_resamp['PROBA_CLUSTER%s'% (cluster)])) 
    metadata['PROBA_CLUSTER%s' % (cluster)] = metadata['pseudoId'].apply(lambda x: d_proba[x] if x in d_proba.keys() else None)

# Color the samples where MTX is missing grey (because we don't use these for the downstream analysis)
metadata.loc[metadata['MTX'] == 0, 'event_1y'] = None
metadata.loc[metadata['MTX'] == 0, 'event'] = None
metadata.loc[metadata['MTX'] == 0, 'time'] = None

metadata.to_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/%s_Stability.csv' % PROJECTION_DATA, index=False)

## ------------------------------------ CREATE 2D Embedding--------------------------------------
l_final_col = [ i for i in metadata.columns if 'PhenoGraph_iter_' not in i ]#[2:]

# Factors
fit1 = TSNE(random_state=SEED+1, n_components=2, perplexity=30, metric='minkowski').fit_transform(Z.values)
fit2 = umap.UMAP(random_state=SEED-1,metric='minkowski').fit(Z.values).embedding_

func.visualize_umap_bokeh_na_2(fit1, metadata[l_final_col], list(l_final_col), l_binary=l_cat, patient_id='sample', cluster_id='PhenoGraph_clusters', title='MMAE_PHENOGRAPH_240_CATEGORIC_STABILITY_TSNE')
func.visualize_umap_bokeh_na_2(fit2, metadata[l_final_col], list(l_final_col), l_binary=l_cat, patient_id='sample', cluster_id='PhenoGraph_clusters', title='MMAE_PHENOGRAPH_240_CATEGORIC_STABILITY_UMAP')


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

plt.savefig("/exports/reum/tdmaarseveen/RA_Clustering/figures/3_clustering/heatmap_unsorted_CATEGORIC.png") 
