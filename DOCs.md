WANDB_NAME=PROB_WOWODB_V1: 2GPUs, BS 2, lr_drop 40 Done Train results is not good.
WANDB_NAME=PROB_WOWODB_V2: 1GPUs, BS 4, lr_drop 40 Break Train results is not good.
Queue Cluster, remember to push the wandb offline
WANDB_NAME=PROB_WOWODB_V3: 2GPUs, BS 4, lr_drop 35 Running

WANDB_NAME=PROB_OWDETR_V1: 2GPUs, BS 4, lr_drop 35 Break Train is likely fine.
WANDB_NAME=PROB_OWDETR_V2: 1GPUs, BS 6, lr_drop 35 Queue





Temporary
python -u main_open_world.py --output_dir exps/MOWODB/PROB/t1 --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --wandb_project --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_V1 --exemplar_replay_cur_file learned_owod_t1_ft.txt