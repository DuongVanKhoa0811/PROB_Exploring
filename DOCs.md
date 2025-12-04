## Servers: KimCuc, PhatDuy, 4_A5000, Cluster, NgocPC, PhatPC, (LeePC)

## Given the setup on one server, how can I set it up on new servers?
git clone <...>
scp -r <...>:<...>/models/dino_resnet50_pretrain.pth ...
scp -r <...>:<...>/data/OWOD/Annotations ...
scp -r <...>:<...>/data/OWOD/JPEGImages ...
install.sh
login wandb


## Replicated results
WANDB_NAME=PROB_MOWODB_Vall: 2GPUs, BS 3               Done Train all task, PROB_MOWODB_Vall/learned_owod_t2_ft.txt contains 08_007409, FileNotFoundError: [Errno 2] No such file or directory: './data/OWOD/JPEGImages/08_007409.jpg'
WANDB_NAME=PROB_OWDETR_Vall: 2GPUs, BS 3               Running Train all task

## Focus task 1
          Configurations                             Status
WANDB_NAME=PROB_MOWODB_V1: 2GPUs, BS 3               Done
WANDB_NAME=PROB_MOWODB_V2: 2GPUs, BS 2               Done
WANDB_NAME=PROB_MOWODB_V3: 2GPUs, BS 4               Done
WANDB_NAME=PROB_MOWODB_V4: 1GPUs, BS 6               Done
WANDB_NAME=PROB_MOWODB_V9: 4GPUs, BS 5               Done
WANDB_NAME=PROB_MOWODB_V10: 8GPUs, BS 5               Done

WANDB_NAME=PROB_OWDETR_V1: 1GPUs, BS 6               Done
WANDB_NAME=PROB_OWDETR_V2: 2GPUs, BS 3               Done


## PROB Extract Obj Features
python -u main_open_world.py --output_dir exps/MOWODB/PROB_V10/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V10/t1.pth --eval --wandb_project '' --batch_size 1


## Visualize the TSNE
