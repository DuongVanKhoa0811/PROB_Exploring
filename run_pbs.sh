#!/bin/bash
#PBS -N PROB
#PBS -q gold
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -l walltime=99:00:00
#PBS -o /home/users/vankhoa_duong/projects/OOD_OD/PROB_Exploring/trash/output.log
#PBS -e /home/users/vankhoa_duong/projects/OOD_OD/PROB_Exploring/trash/err.log

cd /home/users/vankhoa_duong/projects/OOD_OD/PROB_Exploring
source ~/.bashrc  # or source /path/to/conda.sh

# Activate conda environment
conda activate prob

bash run.sh > logs/logs_PROB_SOWODB_Vall.txt 2>&1

# # Set CUDA environment variables if needed
# export CUDA_HOME=$CONDA_PREFIX
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# # Navigate to ops directory and run make.sh
# sh make.sh > a.txt

# Optionally test
# python test.py > b.txt 2>&1