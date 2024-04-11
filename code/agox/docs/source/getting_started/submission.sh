#!/bin/bash
#SBATCH --job-name=rss_script_slurm
#SBATCH --partition=q48,q40,q36,q28,q24
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-50%50

echo "========= Job started  at `date` =========="
echo "My jobid: $SLURM_JOB_ID"
echo "My array id: $SLURM_ARRAY_TASK_ID"

# Source Python environment
source <REPLACE THIS WITH YOUR PYTHON ENVIRONMENT> 

# Move script to scratch: 
cp rss_script_slurm.py /scratch/$SLURM_JOB_ID/.

# Make and move to run subfolder:
mkdir run_$SLURM_ARRAY_TASK_ID
cd run_$SLURM_ARRAY_TASK_ID
MAIN_FOLDER="$(pwd)"

# Move to scratch:
cd /scratch/$SLURM_JOB_ID/

# Execute script:
python rss_script_slurm.py -i $SLURM_ARRAY_TASK_ID 

# Move back to run folder:
mv * $MAIN_FOLDER/.
cd $MAIN_FOLDER

cp db* ../.

echo "========= Job finished at `date` =========="

