#!/bin/bash
set -ex
# export CUDA_VISIBLE_DEVICES=1

GPUS_PER_NODE=2 ./tools/run_dist_launch_eval.sh 2 configs/EVAL_M_OWOD_BENCHMARK_train_obj_1_V18.sh > ./logs/logs_PROB_MOWODB_EVAL_V18_1.txt 2>&1

python -u main_open_world.py --output_dir exps/MOWODB/PROB_V18_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V18_1/t1.pth --eval --wandb_project '' --batch_size 1
