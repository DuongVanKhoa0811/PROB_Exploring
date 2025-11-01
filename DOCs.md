## Servers: KimCuc, PhatDuy, 4_A5000, Cluster, 2_3090


## Focus task 1
          Configurations                             Status
WANDB_NAME=PROB_MOWODB_V1: 2GPUs, BS 3               Done
WANDB_NAME=PROB_MOWODB_V2: 2GPUs, BS 2               
WANDB_NAME=PROB_MOWODB_V3: 2GPUs, BS 4               Done
WANDB_NAME=PROB_MOWODB_V4: 1GPUs, BS 6               Done

WANDB_NAME=PROB_OWDETR_V1: 1GPUs, BS 6               
WANDB_NAME=PROB_OWDETR_V2: 2GPUs, BS 3               Running


## Given the setup on one server, how can I set it up on new servers?
git clone <...>
scp -r <...>:<...>/models/dino_resnet50_pretrain.pth ...
scp -r <...>:<...>/data/OWOD/Annotations ...
scp -r <...>:<...>/data/OWOD/JPEGImages ...
install.sh
login wandb