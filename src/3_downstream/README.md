 -------------------------------------------------------------------------------------------------------------------------------------
| Originally located at EHR-Clustering-RA/src/ | 
 -------------------------------------------------------------------------------------------------------------------------------------
 
 - Bootstrapping.sh
     Bash script that deploys the Bootstrapping.py (within same directory) as a job to run on the SHARK cluster.
     
 - Bootstrapping.py
    Bootstrapping analysis whereby we estimate the stability of clusters by evaluating how often patient co-cluster across 1000 times of   randomly resampled subsets of the data (1000x).
 
 - computation_das.sh
     Bash script that deploys the GPU_Shap.py (within same directory) as a job to run on the SHARK cluster.
 
 - GPU_Shap.py
    Build a surrogate model (XGBoost) to infer the most important features of each cluster in a post-hoc manner. The feature importance is calculated with SHAP, which is a technique adopted from game theory to calculate the relative contribution of each player to the surplus.
    This script is called by the bash job script: computation_das.sh
    Run from : RA_Clustering/
    Command: sbatch src/3_downstream/SHAP_job.sh

-  Compute_DAS.py
    Calculate DAS scores for patient selection across entire follow up
    
    
