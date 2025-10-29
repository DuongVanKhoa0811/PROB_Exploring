#!/bin/bash
#PBS -N PROB_MOWODB_V3
#PBS -q gold
#PBS -l select=1:ncpus=10:ngpus=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o ${PBS_O_WORKDIR}/trash/logs_PROB_MOWODB_V3.log

bash run.sh > ./logs/logs_PROB_MOWODB_V3.txt 2>&1