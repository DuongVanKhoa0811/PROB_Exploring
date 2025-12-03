#!/bin/bash
set -ex

python -u main_open_world.py --output_dir exps/MOWODB/PROB_V10/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V10/t1.pth --eval --wandb_project '' --batch_size 1
