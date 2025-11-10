#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1

GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2 configs/S_OWOD_BENCHMARK_freeze_obj.sh