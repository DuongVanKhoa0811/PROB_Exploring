#!/usr/bin/env bash

echo running training of prob-detr, M-OWODB dataset

set -ex

sleep 10    

EXP_DIR=exps/MOWODB/PROB_V15_1
PY_ARGS=${@:1}
WANDB_NAME=PROB_MOWODB_V15_1
BATCH_SIZE=5

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 51\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    ${PY_ARGS} --batch_size ${BATCH_SIZE} --pretrain exps/MOWODB/PROB_V11/t1/checkpoint.pth
    
sleep 10    

EXP_DIR=exps/MOWODB/PROB_V15_1_seed_52
PY_ARGS=${@:1}
WANDB_NAME=PROB_MOWODB_V15_1_seed_52
BATCH_SIZE=5

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 51\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    ${PY_ARGS} --batch_size ${BATCH_SIZE} --pretrain exps/MOWODB/PROB_V11/t1/checkpoint.pth --seed 52

sleep 10    

EXP_DIR=exps/MOWODB/PROB_V15_1_seed_62
PY_ARGS=${@:1}
WANDB_NAME=PROB_MOWODB_V15_1_seed_62
BATCH_SIZE=5

python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20\
    --train_set 'owod_t1_train' --test_set 'owod_all_task_test' --epochs 51\
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3\
    --wandb_name "${WANDB_NAME}_t1" --exemplar_replay_selection --exemplar_replay_max_length 850\
    --exemplar_replay_dir ${WANDB_NAME} --exemplar_replay_cur_file "learned_owod_t1_ft.txt"\
    ${PY_ARGS} --batch_size ${BATCH_SIZE} --pretrain exps/MOWODB/PROB_V11/t1/checkpoint.pth --seed 62