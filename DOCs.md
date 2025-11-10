## Servers: KimCuc, PhatDuy, 4_A5000, Cluster, NgocPC, PhatPC

## Given the setup on one server, how can I set it up on new servers?
git clone <...>
scp -r <...>:<...>/models/dino_resnet50_pretrain.pth ...
scp -r <...>:<...>/data/OWOD/Annotations ...
scp -r <...>:<...>/data/OWOD/JPEGImages ...
install.sh
login wandb


## Focus task 1
          Configurations                             Status
WANDB_NAME=PROB_MOWODB_V1: 2GPUs, BS 3               Done
WANDB_NAME=PROB_MOWODB_V2: 2GPUs, BS 2               Done
WANDB_NAME=PROB_MOWODB_V3: 2GPUs, BS 4               Done
WANDB_NAME=PROB_MOWODB_V4: 1GPUs, BS 6               Done

WANDB_NAME=PROB_OWDETR_V1: 1GPUs, BS 6               Done
WANDB_NAME=PROB_OWDETR_V2: 2GPUs, BS 3               Done


## exp_obj_train_only branch
WANDB_NAME=PROB_MOWODB_V5: 2GPUs, BS 3               Running
WANDB_NAME=PROB_OWDETR_V3: 2GPUs, BS 3               Running




## Tmp
+ train: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/t1 --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_MOWODB_V1 --exemplar_replay_cur_file learned_owod_t1_ft.txt --batch_size 3

+ eval: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_test/t1.pth --eval
