# EHR-Clustering-RA
Rheumatoid arthritis (RA) is an autoimmune disease that causes swelling and pain in the joints. Currently, patients all receive the same baseline treatment, namely: Methotrexate. The treatment response differs however from patient to patient since RA is a highly heterogeneous disease.  A tantalizing idea is that the factors that cause this latent clinical heterogeneity are already present in patients at baseline.

Therefore, we aim to improve the understanding of RA by "unlocking" the wealth of data available in the Electronic Health Records (EHR). We developed a pipeline that casts different EHR-layers, such as lab- and mannequin- data, into a shared latent space and performs clustering accordingly. Next, we compare the baseline characteristics and treatment response between the clusters.

## Workflow
![alt text](https://github.com/levrex/EHR-Clustering-RA/blob/main/figures/md/fig2_workflow.png?raw=true)
Workflow depicting the different steps of our pipeline. We start off with 25,933 dossiers of patients that visited the outpatient clinic, then we apply a patient selection whereby we request at a minimum follow-up of 1 year and a first visit after the initialization of HIX in 2011. Next, we feed the conclusion section - containing the physician’s verdict - to a Machine Learning Method to predict rheumatoid arthritis (RA) patients. We end up with the records of 928 RA-patients which are supplied to the EHR clustering pipeline, where we preprocess the different EHR-components and combine them in a shared latent space. We employ graph clustering on this feature space to stratify patients. Finally we elucidate the key features driving the clusters and we quantify the association between cluster membership and treatment efficacy.


## Installation

#### Windows systems:
Prerequisite: Install [Anaconda](https://www.anaconda.com/distribution/) with python version 3.8+. This additionally installs the Anaconda Prompt, which you can find in the windows search bar. Use this Anaconda prompt to run the commands mentioned below.

#### Linux / Windows (dev) systems:
Prerequisite: [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) environment (with jupyter notebook). Use the terminal to run the commands mentioned below.

#### Install Jupyter Notebook:
```sh
$ conda install -c anaconda notebook
```

#### Install required modules
Use pip to install the dependecies

```sh
$ pip install -r requirements.txt
```

## Where to start?
- Start a notebook session in the terminal 
- Open the first notebook : 
[Notebook 1](notebooks/1_patient_selection.ipynb)

## File details
* `data/*`: All data is stored here (Excluded: sensitive data)
* `figures/*`: All figures of the project are stored here
* `filters/*`: This directory features different lists reporting the patients eligible for the study (Excluded: sensitive data)
* `models/*`: The Machine learning model used to detect RA-patients is stored here
* `notebooks/*`: Features the pipeline
* `notebooks/1_patient_selection.ipynb`: Applies patient selection steps & machine learning to identify the RA-patients eligible for our study.
* `notebooks/2_extract_features.ipynb`: Notebook whereby we extract the relevant features from the Electronic Health Records 
* `notebooks/3_process_lab.ipynb`: Notebook dedicated to processing the lab data
* `notebooks/4_process_mannequin.ipynb`: Notebook dedicated to processing the mannequin data & creating additional features
* `notebooks/5_validation.ipynb`: Notebook that compares our HIX-extraction to the EAC (Early arthritis cohort) data
* `notebooks/6_fitting_modalities.ipynb`: Notebook that fits the different EHR-modalities and reports them in one big table
* `notebooks/7_clustering.ipynb`: Notebook that subjects the latent feature space from MOFA to perform graph clustering & survival analysis
* `notebooks/8_medication.ipynb`: Notebook that processes the medication information & creates insightful plots
* `results/*`: Stores the MOFA embedding
* `sql/*`: Directory consisting of the different SQL-queries used to extract data from the EHR (Excluded: sensitive data)
* `src/1_emr_scripts/*`: All python scripts used for preprocessing the data are stored here 
* `src/2_imputation_steps/*`: Scripts for imputing the data (we did not impute the data in the current study)
* `src/3_clustering_scripts/*`: Features the R-scripts to run MOFA
* `src/4_downstream/*`: Features scripts for downstream analysis to calculate the SHAP values & the LOESS curve
* `TSNE/*`: The interactive TSNE plots are stored here
