#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1

GPUS_PER_NODE=2 ./tools/run_dist_launch_test.sh 2 configs/M_OWOD_BENCHMARK_test.sh
