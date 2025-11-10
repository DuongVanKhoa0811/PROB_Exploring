## Servers: KimCuc, PhatDuy, 4_A5000, Cluster, NgocPC, PhatPC


## Focus task 1
          Configurations                             Status
WANDB_NAME=PROB_MOWODB_V1: 2GPUs, BS 3               Done
WANDB_NAME=PROB_MOWODB_V2: 2GPUs, BS 2               Done
WANDB_NAME=PROB_MOWODB_V3: 2GPUs, BS 4               Done
WANDB_NAME=PROB_MOWODB_V4: 1GPUs, BS 6               Done

## exp_obj_train_only branch
WANDB_NAME=PROB_MOWODB_V5: 2GPUs, BS 3               Running KimCuc
WANDB_NAME=PROB_OWDETR_V3: 2GPUs, BS 3               Running


## Given the setup on one server, how can I set it up on new servers?
git clone <...>
scp -r <...>:<...>/models/dino_resnet50_pretrain.pth ...
scp -r <...>:<...>/data/OWOD/Annotations ...
scp -r <...>:<...>/data/OWOD/JPEGImages ...
install.sh
login wandb


## Tmp
+ train: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/t1 --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_MOWODB_V1 --exemplar_replay_cur_file learned_owod_t1_ft.txt --batch_size 3
+ train: python -u main_open_world.py --output_dir exps/SOWODB/PROB_V1/t1 --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set owdetr_t1_train --test_set owdetr_test --epochs 41 --lr_drop 31 --model_type prob --obj_loss_coef 1e-3 --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_OWDETR_V1 --exemplar_replay_cur_file learned_owdetr_t1_ft.txt --batch_size 6

+ eval: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_test/t1.pth --eval
