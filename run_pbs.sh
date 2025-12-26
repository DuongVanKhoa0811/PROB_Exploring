#!/bin/bash
#PBS -N PROB_M_V18
#PBS -q gold
#PBS -l select=1:ncpus=32:ngpus=8:mem=256gb
#PBS -l walltime=99:00:00
#PBS -p 00046
#PBS -j oe
#PBS -o /home/users/vankhoa_duong/projects/OOD_OD/PROB_Exploring/logs/output.log

cd /home/users/vankhoa_duong/projects/OOD_OD/PROB_Exploring
source ~/.bashrc
conda activate prob

bash run.sh
