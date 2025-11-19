#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1

set -ex

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK_train_obj_0.sh > logs/logs_PROB_MOWODB_V8_0.txt 2>&1
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK_train_obj_1.sh > logs/logs_PROB_MOWODB_V8_1.txt 2>&1
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 configs/M_OWOD_BENCHMARK_train_obj_2.sh > logs/logs_PROB_MOWODB_V8_2.txt 2>&1