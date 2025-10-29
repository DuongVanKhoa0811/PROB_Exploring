#!/bin/bash
#PBS -N PROB_MOWODB_V3
#PBS -q gold
#PBS -l select=1:ncpus=8:ngpus=2
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -o ${PBS_O_WORKDIR}/trash/logs_PROB_MOWODB_V3.log

bash run.sh > ./logs/logs_PROB_MOWODB_V3.txt 2>&1