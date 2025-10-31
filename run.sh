# #!/bin/bash
export CUDA_VISIBLE_DEVICES=0

GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 configs/M_OWOD_BENCHMARK.sh
