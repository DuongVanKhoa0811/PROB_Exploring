#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/M_OWOD_BENCHMARK.sh