## PROB Extract Obj Features
- python -u main_open_world.py --output_dir exps/MOWODB/PROB_V16_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V16_1/t1.pth --eval --wandb_project '' --batch_size 1


<!-- git commit -m 'Test #3 - Extract penultimate layer features for model define of PROB_MOWODB_V16_1 t1' -->
## Visualize the TSNE
cd models
python prob_features_visualization.py