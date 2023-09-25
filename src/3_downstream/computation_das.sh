#!/bin/bash
#SBATCH -J Compute_DAS
#SBATCH --ntasks=1
#SBATCH -N 1
# SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t.d.maarseveen@lumc.nl

INPUT_DATA='/exports/reum/tdmaarseveen/RA_Clustering/new_data/1_raw/Clustering_Gewrichtspop_with_BSE.csv'
EXPORT_DATA='/exports/reum/tdmaarseveen/RA_Clustering/new_data/7_final/DAS_patients2.csv'
SCHEDULE='/exports/reum/tdmaarseveen/RA_Clustering/new_data/offshoots/DAS_check/DF_REU_Schedule_validate.csv'
METADATA='/exports/reum/tdmaarseveen/RA_Clustering/filters/RA_patients_AllNP_inclTreatmentStart.csv'

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
conda activate ra_clustering2 #../convae_architecture/envs

# Run job
#python src/3_downstream/CalculateDAS_alternate.py
python src/3_downstream/Compute_DAS.py $INPUT_DATA $EXPORT_DATA $SCHEDULE $METADATA
# TEST : python Multimodal_autoencoder_test.py

echo "Program finished with exit code $? at: `date`"