#!/bin/bash
set -ex
# export CUDA_VISIBLE_DEVICES=1

GPUS_PER_NODE=2 ./tools/run_dist_launch_eval.sh 2 configs/EVAL_M_OWOD_BENCHMARK_V7.sh > ./logs/logs_PROB_MOWODB_EVAL_V7.txt 2>&1