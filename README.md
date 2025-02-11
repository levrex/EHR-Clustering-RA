# EHR-Clustering-RA
Rheumatoid arthritis (RA) is an autoimmune disease that causes swelling and pain in the joints. As others, we reckon that RA is an heterogeneous disease and that subset identification is an important first step to enable targeted therapy and a better etiologic understanding. A tantalizing idea is that the factors that cause this latent clinical heterogeneity are already present in patients at baseline.

Therefore, we aim to refine the taxonomy of RA by "unlocking" the wealth of data available in the Electronic Health Records (EHR). We developed a pipeline that casts different EHR-layers, such as lab- and clinical- data, into a shared latent space and performs clustering accordingly. Next, we compare the baseline characteristics and treatment response between the clusters.


## Workflow
![alt text](https://github.com/levrex/EHR-Clustering-RA/blob/main/figures/md/fig2_workflow.png?raw=true)

Our study consists of three distinct phases: i) developmental phase where we identify and validate subtypes in a discovery cohort (set A) based on long-term outcomes. ii) replication phase where cluster new patients using historical trial data (Set B) and external hospital data (Set C) to assess generalizability by replicating the treatment analysis, iii) downstream analysis where we examine differences between clusters in synovial tissue using external hospital data (set D).

We preprocess  different EHR-components and combine this information to construct a patient embedding with MMAE. We employ graph clustering (w/ PhenoGraph) on this feature space to stratify patients. The clusters were further analyzed with a SHAP analysis, to identify the signatures that drive the distinct clinical manifestions

### Preprint
Note: we used this pipeline for our study, of which you can find a preprint here: https://doi.org/10.1101/2023.09.19.23295482. 

### Joint involvement patterns (JIP)
We identified four different RA subsets, characterized mainly by their Joint Involvement Patterns (**JIP**): JIP-foot) arthritis in feet, JIP-oligo) seropositive oligo-articular disease, JIP-hand) seronegative hand arthritis, JIP-poly) polyarthritis. We conducted sensitivity analysis, external validation and 1000 times bootstrapping to ensure that our clusters were valid, generalizable and stable.

![alt text](https://github.com/levrex/EHR-Clustering-RA/blob/main/figures/md/fig3_SHAP_ClusterOverview.png?raw=true)

To ensure the robustness of our findings, we tested a) cluster stability (1000 fold) b) physician confounding, c) association with remission and methotrexate failure, d) generalizability to independent data (historic trial n=307; external hospitals n=515), e) association with histopathological features (SYNgem cohort n=194).

## Webtool
This project provides a client-based webtool tool that classifies rheumatoid arthritis into four distinct phenotypes based on initial clinical presentation, that were primarily characterized by their unique joint involvement pattern (JIP): feet, oligoarticular, hand, and polyarticular distribution (credit: Nick Bos)
[Click here to start clustering](https://knevel-lab.github.io/Rheumalyze/)

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
* `embedding/*`: Here you'll find the patient embedding created with MMAE, based on set A
* `figures/*`: All figures of the project are stored here
* `filters/*`: This directory features different lists reporting the patients eligible for the study (Excluded: sensitive data)
* `models/*`: The Machine learning model used to detect RA-patients is stored here
* `notebooks/*`: Features the entire pipeline divided in steps over different notebooks.
* `notebooks/1_patient_selection.ipynb`: Employs machine learning to identify the diagnosis of the physician - to collect our RA-patients.
* `notebooks/2_extract_features.ipynb`: Notebook whereby we extract the relevant features from the Electronic Health Records 
* `notebooks/3_process_lab.ipynb`: Notebook dedicated to processing the lab data from the EHR
* `notebooks/4_process_mannequin.ipynb`: Notebook dedicated to processing the joint mannequin data from the EHR & creating additional features
* `notebooks/5_clustering.ipynb`: This notebook performs PhenoGraph graph clustering on the patient embedding created with MMAE and 
    creates a metadata table. Several steps are taken to define the ideal number of clusters and to 
    evaluate the robustness of the clustering technique.
* `notebooks/6_process_medication.ipynb`: Notebook that process the medication information
* `notebooks/7_survival_analysis.ipynb`: Notebook that performs survival analysis to see if our clusters correspond to long term outcomes (MTX-discontinuation, Remission)
* `notebooks/8_replication_setB_IMPROVED.ipynb`: Notebook that processes & projects replication data from set B (historic trial). To see if our clusters are recurring in an independent data set.
* `notebooks/8_replication_setB_RZWN.ipynb`: Notebook that processes & projects replication data from set C (external hospitals). To see if our clusters are recurring in independent data.
* `notebooks/9_treatment_analysis.ipynb`: Notebook that checks for any noticeable differences in treatment response.
* `notebooks/10_build_MTX_predictor.rmd`: R Markdown notebook to perform downstream analysis steps like: checking for Physican bias, global trend in treatment response and evaluating the quality of an MTX-success predictor 
* `notebooks/11_Downstream_histopathological_analysis.ipynb`: Notebook that processes & projects data from set D (synovial tissue data from SYNGem cohort) to examine if clusters differ in histopathological features.
* `notebooks/additional_steps/Compute_DAS.ipynb*`: Computes the disease activity scores based on the components, to counter the otherwise large missingness
* `results/treatmentRespons`: Stores the OR for the different patients.
* `sql/*`: Directory consisting of the different SQL-queries used to extract data from the Leiden EHR digital system (Excluded: sensitive data)
* `src/1_emr_scripts/*`: All python scripts used for preprocessing the data are stored here 
* `src/2_latent_space/*`: Scripts for creating the patient embedding with MMAE
* `src/3_downstream/*`: Features scripts for downstream analysis like Bootstrapping or SHAP experiment
* `TSNE/*`: The interactive TSNE plots are stored here (Blinded)

### Replication in a second dataset
We replicated the clustering and the downstream analysis (SHAP & survival analysis) in seperate independent datasets for external validation. To investigate the replicability of the clustering we projected our novel patients unto our learned patient embedding, and assigned them to a cluster based on their orientation with respect to the old data. This projection was achieved with [POODLE](https://github.com/levrex/Poodle)

## Contact
If you experience difficulties with implementing the pipeline or if you have any other questions feel free to send me an e-mail. You can contact me on: t.d.maarseveen@lumc.nl
