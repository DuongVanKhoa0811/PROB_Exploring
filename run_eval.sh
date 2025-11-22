#!/bin/bash
set -ex
# export CUDA_VISIBLE_DEVICES=2

GPUS_PER_NODE=2 ./tools/run_dist_launch_eval.sh 2 configs/EVAL_M_OWOD_BENCHMARK.sh > ./logs/logs_PROB_MOWODB_EVAL_V9.txt 2>&1