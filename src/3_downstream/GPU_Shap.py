import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import seaborn as sn
import shap
import numpy as np
from sklearn.model_selection import KFold
from matplotlib.colors import LinearSegmentedColormap # for custom palette
from matplotlib import colors as plt_colors
import sys
from scipy import spatial

import sys
sys.path.append(r'src/1_emr_scripts')
import MannequinFunctions as func

INPUT_FILE = sys.argv[1]
ONLY_CATEGORIC = sys.argv[2]
ONLY_NUMERIC = sys.argv[3]

print('INPUT_FILE=', INPUT_FILE, '; ONLY_CATEGORIC=', ONLY_CATEGORIC, '; ONLY_NUMERIC=', ONLY_NUMERIC)

# Get most important features
def global_shap_importance(model, X):
    """ Return a dataframe containing the features sorted by Shap importance
    Parameters
    ----------
    model : The tree-based model 
    X : pd.Dataframe
         training set/test set/the whole dataset ... (without the label)
    Returns
    -------
    pd.Dataframe
        A dataframe containing the features sorted by Shap importance
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance

# Preload function
def strip_right(df, suffix='_positive'):
    df.columns = df.columns.str.replace(suffix+'$', '', regex=True)


# Get Metadata

## Numerical
df_counts = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/5_clustering/df_mannequin_counts_scaled.csv', sep='|') 
df_numeric = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/5_clustering/df_lab_raw_demographics.csv', sep=',')[['Leuko', 'MCH', 'Hb', 'Ht', 'MCHC',  'MCV', 'Trom', 'BSE',  'Age']] # 'Lym', 'Mono',s

# df_lab_scaled_demographics.csv'

df_num = df_numeric #df_counts.merge(df_numeric, left_index=True, right_index=True)

l_num = list(df_num.columns)

# Categorical data
df_cat = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/new_data/5_clustering/df_categoric.csv') # df_categoric['Age_Early']
l_cat = list(df_cat.columns)
l_cat = [col for col in l_cat if 'negative' not in col]
l_cat = [i for i in l_cat if i not in ['PHYSICIAN', 'pseudoId']]  # , 'Age_Early'

## Merge Categorical + Numerical
df_imp = df_cat[l_cat].merge(df_num, left_index=True, right_index=True)
strip_right(df_imp) 



# Get MAUI embedding
df_patient = pd.read_csv(INPUT_FILE) 
cluster_id = 'PhenoGraph_clusters'
sample_id = 'sample'
technique = 'MMAE'

# Get SEURAT embedding
#df_patient = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/results/seurat/patient_clusterids3.csv') 
#cluster_id = 'clusterid_0.5'
#technique = 'seurat'
# df_patient[cluster_id] = df_patient[cluster_id].astype(int)
# df_patient[cluster_id] = df_patient[cluster_id] - 1



# Combine embedding with metadata
df_imp = df_imp.merge(df_patient[[cluster_id, sample_id]], left_index=True, right_index=True) # 'PHYSICIAN',  'clusterid_0'

# Dependent variable = cluster membership
if ONLY_CATEGORIC == ONLY_NUMERIC: 
    ## Use all relevant columns to calculate SHAP impact values
    cols_data = [x for x in list(df_imp.columns) if x not in ['pseudoId', 'patnr', 'sample', 'time', 'coor_x', 'coor_y', cluster_id, 'Physician', 'prediction','Lookahead_Treatment', 'Lookbehind_Treatment', 'FirstConsult', 'birthDate', 'FirstDAS', 'firstMannequin', 'DAS28(3)', 'DAS44']] #   'Sex', 'Age', 
elif ONLY_CATEGORIC=="1":
    cols_data = [i[:-9] if i.endswith('_positive') else i for i in l_cat ]
elif ONLY_NUMERIC=="1":
    cols_data = l_num


    
# Define X (predictors) and y (label/ dependent var)
X = df_imp[cols_data].values
y = df_imp[cluster_id]
pid = df_imp[sample_id]#.values


# Apply 5 fold CV
kf = KFold(n_splits=5) # , random_state=0
iteration = 0
y_pred = []

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index[:10], "TEST:", test_index[:10])
    X_train, X_test_raw = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pid_train, pid_test = pid[train_index], pid[test_index]

    # Normalize
    fit_gaussian = False

    # Z-score scaling
    scaler = StandardScaler().fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test_raw)

    # Model is an XGBClassifier
    n_trees = 500
    dmat_train = xgb.DMatrix(X_train, y_train)
    dmat_test = xgb.DMatrix(X_test, y_test)

    t0 = time.time()
    bst  = xgb.train({'objective': 'multi:softmax', 'num_class':len(y.unique())}, dmat_train,
                        n_trees, evals=[(dmat_train, "train"), (dmat_test, "test")]) # , early_stopping_rounds=10 , early_stopping_rounds=100
    t1 = time.time()
    print('Time for Training XGB model %s: %s' % (str(iteration+1), str(t1-t0)))
    iteration += 1
    
    # Create a confusion matrix over all data!
    y_pred.extend(bst.predict(dmat_test))

# Make sure GPU prediction is enabled
# bst.set_param({"predictor": "gpu_predictor"})

# Make confusion table
fig = plt.figure()
cm = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)


df_cm = pd.DataFrame(cm, index = list(range(len(y.unique()))),
                  columns = list(range(len(y.unique()))))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel("Posthoc XGB-predictions")
plt.ylabel("Actual Cluster-labels")
plt.title('Confusion matrix for the XGBoost classifier - both categorical + numerical (ACC: %.2f)' % (accuracy))
plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/%s_confusion_matrix.png' % (technique), dpi=100)

# Compute the shap values
t0 = time.time()
#shap_values = bst.predict(dmat_test, pred_contribs=True)
t_explainer = shap.TreeExplainer(bst) # or just X?
shap_values = t_explainer.shap_values(X_test)
t1 = time.time()
print('Calculating SHAP: ' + str(t1-t0))

plt.clf()

# Make custom palette same as tSNE
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))

l_hex = ["#1578EC", "#ff9940", "#31D631", "#dd4d4e", "#7a4da4", "#a67c73"]
l_rgb = [hex2rgb(i) for i in l_hex]


# get class ordering from shap values
class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])

# create listed colormap
cmap = plt_colors.ListedColormap(np.array(l_rgb)[class_inds])

#shap.plots.heatmap(shap_values)
#df_shap = global_shap_importance(bst, X_test)
vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(cols_data,sum(vals))),columns=['col_name','feature_importance_vals'])
#print(feature_importance)
feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
#feature_importance.head()

#print(shap_values[1])
N_FEAT = 15

feature_names = feature_importance['col_name'].values[:N_FEAT]
df_shapley = pd.DataFrame(data=X_test, columns = cols_data)
X_test_filt = df_shapley[feature_names].values



for class_idx in range(len(y.unique())):
    plt.figure(figsize=(5, 8)) # , dpi=80
    shap.summary_plot(pd.DataFrame(data=shap_values[class_idx], columns = cols_data)[feature_names].values, X_test_filt, feature_names=feature_names, show=False, plot_type="bar",sort=False) # or just X
    # color=cmap[class_idx],
    
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_class%s_Multi_n%s.png' % (str(class_idx), N_FEAT))
    plt.clf()

# Adjusted labels - to show weight
for class_idx in range(len(y.unique())):
    # Get top 15 of current set
    vals= np.abs([shap_values[class_idx]]).mean(0)
    feature_importance = pd.DataFrame(list(zip(cols_data,sum(vals))),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_names = feature_importance['col_name'].values[:N_FEAT]

    plt.figure(figsize=(2, 8)) # , dpi=80
    shap.summary_plot(pd.DataFrame(data=shap_values[class_idx], columns = cols_data)[feature_names].values, df_shapley[feature_names].values, feature_names=feature_names, show=False, sort=False) # or just X
    # color=cmap[class_idx],
    
    fig, ax = plt.gcf(), plt.gca()
    ax.set_yticks(ax.get_yticks().tolist())
    # 20 ?
    ax.set_yticklabels(['%s %s' % (list(range(N_FEAT)[::-1])[ix]+1, func.rename_mannequin_features(i)) for ix, i in enumerate(feature_names[::-1])], fontsize=23) # not sure if we need to reverse this
    
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_class%s_n%s.png' % (str(class_idx), N_FEAT))
    plt.clf()
    
shap.summary_plot(shap_values, X_test, cols_data, show=False,color=cmap, class_names=list(range(len(y.unique())))) # or just X
plt.tight_layout()
plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/maui_shap_summary_4clusters.png')
            
print('use palette')
#plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/maui_shap_heatmap.png', bbox_inches='tight')

#plt.tight_layout()
plt.clf()


# Plot each class
#for class_idx in range(len(y.unique())):
#    fig = shap.summary_plot(shap_values[class_idx], X_test, cols_data, show=False, max_display=15) # default = 20
#    plt.subplots_adjust(bottom=0.4)
#    plt.tight_layout()
#    plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_class%s.png' % (str(class_idx)))
#    plt.clf()

# Force plot
for class_idx in range(len(y.unique())):
    fig = shap.force_plot(t_explainer.expected_value[class_idx], shap_values[class_idx], X_test, cols_data)
    shap.save_html('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Patients_class%s.html' % (str(class_idx)), fig)
    plt.clf()
    
# Raw Shap weights
for class_idx in range(len(y.unique())):
    try: 
        np.array(shap_values[0])
    except:
        print(shap_values[0])
        print(eql)
    #df_shap = pd.DataFrame(list(np.array(shap_values[class_idx])), columns=cols_data)
    #df_shap.to_csv('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_raw_class%s.csv' % (str(class_idx)), sep='|', index=False)
    vals= np.abs([shap_values[class_idx]]).mean(0)

    feature_importance = pd.DataFrame(list(zip(cols_data,sum(vals))),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)

    feature_importance.to_csv('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/SHAP_RANKED_raw_class%s.csv' % (str(class_idx)), sep='|', index=False)


# Save force plot for each archetypal patient
df_arch = pd.DataFrame(X_test, columns=cols_data)
df_arch['PhenoGraph_clusters'] = y_test.values
df_arch[sample_id] = pid_test.values
print(df_arch.head())
#df_arch['cluster'].unique()

l_arch = []
for class_idx in range(4):
    indices = df_arch[df_arch['PhenoGraph_clusters']==class_idx].index
    sub_df = df_arch[df_arch['PhenoGraph_clusters']==class_idx].copy()
    cluster_center = sub_df.median()
    l_sim = []
    for ix, row in sub_df.iterrows():
        l_sim.append(spatial.distance.minkowski(cluster_center, row))
    print(l_sim)
    #print(min(l_sim), indices)
    l_arch.append(indices[np.argmin(l_sim)])
    #l_arch.append(indices[np.argpartition(l_sim, 3)[1]]) # Take a random out of 3 archetypal patients?
    
    #l_arch.append(indices[np.argmin(l_sim)])
    archetypal_patient = indices[np.argmin(l_sim)]
    #archetypal_patient =  indices[np.argpartition(l_sim, 3)[1]]  # Take a random out of 3 archetypal patients?
    
    archetypal_pid = sub_df.loc[archetypal_patient][sample_id]
    
    fig = shap.waterfall_plot(shap.Explanation(values=shap_values[class_idx][archetypal_patient], 
                                             base_values=t_explainer.expected_value[class_idx], 
                                             data=X_test_raw[archetypal_patient],  
                                             feature_names=cols_data), max_display= N_FEAT
                           )
    
    #fig = shap.plots._waterfall.waterfall_legacy(t_explainer.expected_value[class_idx], shap_values[class_idx][archetypal_patient], feature_names =cols_data,  max_display = N_FEAT,  show=False, NEGATIVE_COLOR='dimgray')
    #fig = shap.force_plot(t_explainer.expected_value[class_idx], shap_values[class_idx][archetypal_patient], X_test[archetypal_patient], cols_data)
    plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Cluster_%s_Patient_%s.png' % (class_idx, archetypal_pid), bbox_inches="tight")
#    shap.save_html('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Cluster_%s_Patient_%s.html' % (class_idx, archetypal_patient), fig)
    plt.clf()
    
# Save individual plot of patient 0 from the test set
#j = 0
#fig = shap.force_plot(t_explainer.expected_value[class_idx], shap_values[class_idx][j], X_test[j], cols_data)
#shap.save_html('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Patient_1.html', fig)
#plt.clf()

# Dependence plot
shap_interaction_values = t_explainer.shap_interaction_values(X_test)

#shap.dependence_plot(
#    ("Age", "Big Joints"),
#    shap_interaction_values[0], X_test,
#    display_features=cols_data
#)
#plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_dependence_Age_bigJoints.png'#)


# Assess added value of categorical data by generating a heatmap that only has numerical lab information
X = df_imp[list(df_numeric.columns)].values  
y = df_imp[cluster_id]


# Apply 5 fold CV
kf = KFold(n_splits=5) # , random_state=0
iteration = 0
y_pred = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize
    fit_gaussian = False

    # Z-score scaling
    scaler = StandardScaler().fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Model is an XGBClassifier
    n_trees = 500
    dmat_train = xgb.DMatrix(X_train, y_train)
    dmat_test = xgb.DMatrix(X_test, y_test)

    t0 = time.time()
    bst  = xgb.train({'objective': 'multi:softmax', 'num_class':len(y.unique())}, dmat_train,
                        n_trees, evals=[(dmat_train, "train"), (dmat_test, "test")]) # "tree_method": "gpu_hist", 
    t1 = time.time()
    print('Time for Training XGB model %s: %s' % (str(iteration+1), str(t1-t0)))
    iteration += 1
    
    # Create a confusion matrix over all data!
    y_pred.extend(bst.predict(dmat_test))

# Make confusion table
fig = plt.figure()
cm = confusion_matrix(y, y_pred)
title = 'Confusion matrix for the XGBoost classifier - only numerical'
df_cm = pd.DataFrame(cm, index = list(range(len(y.unique()))),
                  columns = list(range(len(y.unique()))))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel("posthoc XGB-predictions")
plt.ylabel("Actual Cluster-labels")
plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/%s_confusion_matrix_numerical.png' % (technique), dpi=100)


# Assess added value of categorical data by generating a heatmap that only considers the mannequin counts
#X = df_imp[list(df_counts.columns)].values  
#y = df_imp[cluster_id]

# Apply 5 fold CV
#kf = KFold(n_splits=5) # , random_state=0
#iteration = 0
#y_pred = []

#for train_index, test_index in kf.split(X):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#
#    # Normalize
#    fit_gaussian = False
#
#    # Z-score scaling
#    scaler = StandardScaler().fit(X_train)
#    X_train= scaler.transform(X_train)
#    X_test = scaler.transform(X_test)
#
#    # Model is an XGBClassifier
#
#    n_trees = 500
#    dmat_train = xgb.DMatrix(X_train, y_train)
#    dmat_test = xgb.DMatrix(X_test, y_test)
#
#    t0 = time.time()
#    bst  = xgb.train({'objective': 'multi:softmax', 'num_class':len(y.unique())}, dmat_train,
#                        n_trees, evals=[(dmat_train, "train"), (dmat_test, "test")]) # "tree_method": "gpu_hist", 
#    t1 = time.time()
#    print('Time for Training XGB model %s: %s' % (str(iteration+1), str(t1-t0)))
#    iteration += 1
#    
#    # Create a confusion matrix over all data!
#    y_pred.extend(bst.predict(dmat_test))

# Make confusion table
#fig = plt.figure()
#cm = confusion_matrix(y, y_pred)
#title = 'Confusion matrix for the XGBoost classifier - only counts'
#df_cm = pd.DataFrame(cm, index = list(range(len(y.unique()))),
#                  columns = list(range(len(y.unique()))))
#plt.figure(figsize = (10,7))
#sn.heatmap(df_cm, annot=True, fmt='g')
#plt.xlabel("posthoc XGB-predictions")
#plt.ylabel("Actual Cluster-labels")
#plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/%s_confusion_matrix_counts.png' % (technique), dpi=100)

# Assess how much information can be explained by using categorical data only

#def strip_list(l_cat, suffix='_positive'):
#    df.columns = [i.replace(suffix+'$', '', regex=True) for i in l_cat]

cols_cat = [i[:-9] if i.endswith('_positive') else i for i in l_cat ]
print('FINAL CATEGORIES:', cols_cat)

X = df_imp[cols_cat].values
y = df_imp[cluster_id]


# Apply 5 fold CV
kf = KFold(n_splits=5) # , random_state=0
iteration = 0
y_pred = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Normalize
    fit_gaussian = False

    # Z-score scaling
    scaler = StandardScaler().fit(X_train)
    X_train= scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Model is an XGBClassifier

    n_trees = 500
    dmat_train = xgb.DMatrix(X_train, y_train)
    dmat_test = xgb.DMatrix(X_test, y_test)

    t0 = time.time()
    bst  = xgb.train({'objective': 'multi:softmax', 'num_class':len(y.unique())}, dmat_train,
                        n_trees, evals=[(dmat_train, "train"), (dmat_test, "test")]) # "tree_method": "gpu_hist", 
    t1 = time.time()
    print('Time for Training XGB model %s: %s' % (str(iteration+1), str(t1-t0)))
    iteration += 1
    
    # Create a confusion matrix over all data!
    y_pred.extend(bst.predict(dmat_test))

# Make sure GPU prediction is enabled
# bst.set_param({"predictor": "gpu_predictor"})

# Make confusion table
fig = plt.figure()
cm = confusion_matrix(y, y_pred)
title = 'Confusion matrix for the XGBoost classifier - only categorical'
df_cm = pd.DataFrame(cm, index = list(range(len(y.unique()))),
                  columns = list(range(len(y.unique()))))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel("posthoc XGB-predictions")
plt.ylabel("Actual Cluster-labels")
plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/%s_confusion_matrix_categorical.png' % (technique), dpi=100)