#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/M_OWOD_BENCHMARK.sh > logs/logs_PROB_MOWODB_V11.txt 2>&1