# import os
# import shutil
# import wandb


# wandb.init(project='a', entity="marvl", name='a')


from collections import deque

def print_last_n_lines(filename, n=10):
    with open(filename, 'r') as f:
        last_lines = deque(f, maxlen=n)
    
    for line in last_lines:
        print(line, end='')

# Usage
print_last_n_lines('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/logs_0_tmp.txt', 100)