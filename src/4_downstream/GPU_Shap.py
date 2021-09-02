import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import time
import seaborn as sn
import shap
import numpy as np

# Preload function
def strip_right(df, suffix='_positive'):
    df.columns = df.columns.str.replace(suffix+'$', '', regex=True)

# Get embedding
df_patient = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/8_final/all_data_Mofa_clustering.csv')


# Get Metadata
## Numerical
df_num = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_all_numerical.csv', sep=',')
l_num = list(df_num.columns)

## Categorical
df_e = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_lab_serology.csv')
df_a = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_mannequin_categorical_ohe.csv')

df_cat = df_e.merge(df_a, left_index=True, right_index=True)
l_cat = list(df_cat.columns)
l_cat = [col for col in l_cat if 'negative' not in col]
l_cat = [i for i in l_cat if i not in ['PHYSICIAN']]

## Merge Categorical + Numerical
df_imp = df_cat[l_cat].merge(df_num, left_index=True, right_index=True)
strip_right(df_imp) 

## Use all relevant columns to calculate SHAP impact values
cols_data = [x for x in list(df_imp.columns) if x not in ['patnr', 'time', 'coor_x', 'coor_y', 'PhenoGraph_clusters', 'Physician', 'PHYSICIAN', 'prediction', 'Age', 'Sex']]

# Combine embedding with metadata
df_imp = df_imp.merge(df_patient[['patnr', 'prediction', 'PhenoGraph_clusters']], left_index=True, right_index=True) # 'PHYSICIAN', 

# Dependent variable = cluster membership
data = df_imp[cols_data].values
target = df_imp['PhenoGraph_clusters']

print(len(data), len(target))

# Create Train & Test set
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                   )
print("Training records: {}".format(X_train.shape[0]))
print("Testing records: {}".format(X_test.shape[0]))

# Normalize
fit_gaussian = False

# Z-score scaling
scaler = StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test = scaler.transform(X_test)

if fit_gaussian:
    pt = PowerTransformer().fit(X_train)
    X_train = pt.transform(X_train)
    X_test = pt.transform(X_test)

# Train classifier
#bst.predict(X_test)

# Model is an XGBClassifier

n_trees = 500
dmat_train = xgb.DMatrix(X_train, y_train)
dmat_test = xgb.DMatrix(X_test, y_test)

t0 = time.time()
bst  = xgb.train({"tree_method": "gpu_hist", 'objective': 'multi:softmax', 'num_class':len(target.unique())}, dmat_train,
                    n_trees, evals=[(dmat_train, "train"), (dmat_test, "test")])
t1 = time.time()
print('Time for Training XGB model: ' + str(t1-t0))

# Make sure GPU prediction is enabled
bst.set_param({"predictor": "gpu_predictor"})

# Make confusion table
fig = plt.figure()
y_pred = bst.predict(dmat_test)
cm = confusion_matrix(y_test, y_pred)
title = 'Confusion matrix for the XGBoost classifier'
df_cm = pd.DataFrame(cm, index = list(range(len(target.unique()))),
                  columns = list(range(len(target.unique()))))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mofa_confusion_matrix.png', dpi=100)

# Compute the shap values
t0 = time.time()
#shap_values = bst.predict(dmat_test, pred_contribs=True)
t_explainer = shap.TreeExplainer(bst) # or just X?
shap_values = t_explainer.shap_values(X_test)
t1 = time.time()
print('Calculating SHAP: ' + str(t1-t0))

plt.clf()
# and plot
fig = shap.summary_plot(shap_values, X_test, cols_data, show=False) # or just X
plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mofa_shap_summary.png')
plt.subplots_adjust(left=0.4)
plt.tight_layout()
plt.clf()

# Plot each individual 
for class_idx in range(len(target.unique())):
    fig = shap.summary_plot(shap_values[class_idx], X_test, cols_data, show=False) # or just X
    plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_class%s.png' % (str(class_idx)))
    plt.subplots_adjust(bottom=0.4)
    plt.tight_layout()
    plt.clf()

# Force plot
for class_idx in range(len(target.unique())):
    fig = shap.force_plot(t_explainer.expected_value[class_idx], shap_values[class_idx], X_test, cols_data)
    shap.save_html('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Patients_class%s.html' % (str(class_idx)), fig)
    plt.clf()
    
# Raw Shap weights
for class_idx in range(len(target.unique())):
    try: 
        np.array(shap_values[0])
    except:
        print(shap_values[0])
        print(eql)
    df_shap = pd.DataFrame(list(np.array(shap_values[class_idx])), columns=cols_data)
    df_shap.to_csv('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_shap_raw_class%s.csv' % (str(class_idx)), sep='|', index=False)
    
# Save individual plot of patient 0 from the test set
j = 0
fig = shap.force_plot(t_explainer.expected_value[class_idx], shap_values[class_idx][j], X_test[j], cols_data)
shap.save_html('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_force_plot_Patient_1.html', fig)
plt.clf()

# Dependence plot
shap_interaction_values = t_explainer.shap_interaction_values(X_test)

#shap.dependence_plot(
#    ("Age", "Big Joints"),
#    shap_interaction_values[0], X_test,
#    display_features=cols_data
#)
#plt.savefig('/exports/reum/tdmaarseveen/RA_Clustering/figures/4_downstream/mannequin_dependence_Age_bigJoints.png'#)