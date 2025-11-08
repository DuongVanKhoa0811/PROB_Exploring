#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=1

GPUS_PER_NODE=1 ./tools/run_dist_launch_eval.sh 1 configs/EVAL_M_OWOD_BENCHMARK_V1.sh > ./logs/logs_PROB_MOWODB_EVAL_V1.txt 2>&1