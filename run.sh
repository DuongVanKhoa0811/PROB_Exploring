# #!/bin/bash
export CUDA_VISIBLE_DEVICES=1
GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 configs/S_OWOD_BENCHMARK.sh
