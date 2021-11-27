#!/bin/bash
#SBATCH --job-name=exp_run
#SBATCH --output=./Logs/SLURM/R_%j_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32Gb
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 4 
#SBATCH -p unkillable 

module load anaconda/3
conda activate nsv3

#python test.py 
echo 'running experiment' + $2
echo $2 > ./Logs/SLURM/R_%j_output.txt
python -u train_wrapper.py $2 
