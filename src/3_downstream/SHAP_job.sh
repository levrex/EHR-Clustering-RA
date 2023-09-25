#!/bin/bash
#SBATCH -J SHAP_example
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t.d.maarseveen@lumc.nl

# all_data_MAUI_clustering_90_CATEGORIC / all_data_MAUI_clustering_90_2
INPUT_FILE='/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/MMAE_clustering_270.csv' # all_data_MAUI_clustering_240_NEW # _PsA
ONLY_CATEGORIC="0"
ONLY_NUMERIC="0"

module purge

# Load Cuda & Conda
module add library/cuda/11.2/gcc.8.3.1
module load tools/miniconda/python3.8/4.9.2

echo "Starting at `date`"

echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_JOB_NUM_NODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node running script: $SLURMD_NODENAME"
echo "Submit host: $SLURM_SUBMIT_HOST"

echo "Current working directory is `pwd`"

# Load custom environment
conda activate ../convae_architecture/envs

# Run job
python src/3_downstream/GPU_Shap.py $INPUT_FILE $ONLY_CATEGORIC $ONLY_NUMERIC
# TEST : python Multimodal_autoencoder_test.py

echo "Program finished with exit code $? at: `date`"