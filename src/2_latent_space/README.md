 -------------------------------------------------------------------------------------------------------------------------------------
| Originally located at EHR-Clustering-RA/src/2_latent_space | 
 -------------------------------------------------------------------------------------------------------------------------------------
 
- MMAE_job_shark.sh
     Bash script that deploys the MMAE_multilayered.py (within same directory) as a job to run on the SHARK cluster.
     Run from within the RA_Clustering/src/2_latent_space : sbatch MMAE_job_shark.sh  
     
- MMAE_multilayered.py
    Fit an autoencoder (MMAE) to learn latent factors from multi-modal EHR-data. 
    This script is called by the bash job script: MMAE_job_shark.sh
    
- MMAE_singleLayer.py
    Fit an autoencoder (MMAE) to learn latent factors from a single layer of EHR-data (e.g. only numerical lab or only categorical clinical features). 
    This script is called by the bash job script: MMAE_job_shark.sh

