# Extract boxes with IoU > 0.8

- Draw the predicted boxes on the image
- Draw the gt boxes on the image
- Compute the IoU
- Extract object features

# PROB Extract Obj Features

python -u main_open_world.py --output_dir exps/MOWODB/PROB_V18_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V18_1/t1.pth --eval --wandb_project '' --batch_size 1

python -u main_open_world.py --output_dir exps/MOWODB/PROB_V18_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 157.3 --pretrain exps/MOWODB/PROB_V18_1/t1.pth --eval --wandb_project '' --batch_size 1

# Visualize the TSNE
cd models
python prob_features_visualization.py

# Tmp
+ train: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/t1 --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 41 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_MOWODB_V1 --exemplar_replay_cur_file learned_owod_t1_ft.txt --batch_size 1
+ train: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/t2 --dataset TOWOD --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 20 --train_set owod_t2_train --test_set 'owod_all_task_test' --epochs 51 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model --exemplar_replay_selection --exemplar_replay_max_length 1743 --exemplar_replay_dir PROB_MOWODB_test --exemplar_replay_prev_file learned_owod_t1_ft.txt --exemplar_replay_cur_file learned_owod_t2_ft.txt --pretrain exps/MOWODB/PROB_test/t1/checkpoint0040.pth --lr 2e-5 --batch_size 1
+ train: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/t3 --dataset TOWOD --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set owod_t3_train --test_set owod_all_task_test --epochs 121 --model_type prob --obj_loss_coef 8e-4 --freeze_prob_model --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 2361 --exemplar_replay_dir PROB_MOWODB_test --exemplar_replay_prev_file learned_owod_t2_ft.txt --exemplar_replay_cur_file learned_owod_t3_ft.txt --pretrain exps/MOWODB/PROB_test/t2_ft/checkpoint0110.pth --lr 2e-5 --batch_size 1

    
+ train: python -u main_open_world.py --output_dir exps/SOWODB/PROB_V1/t1 --dataset OWDETR --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set owdetr_t1_train --test_set owdetr_test --epochs 41 --lr_drop 31 --model_type prob --obj_loss_coef 1e-3 --obj_temp 1.3 --exemplar_replay_selection --exemplar_replay_max_length 850 --exemplar_replay_dir PROB_OWDETR_V1 --exemplar_replay_cur_file learned_owdetr_t1_ft.txt --batch_size 6

+ eval: python -u main_open_world.py --output_dir exps/MOWODB/PROB_test/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_test/t1.pth --eval --batch_size 1
