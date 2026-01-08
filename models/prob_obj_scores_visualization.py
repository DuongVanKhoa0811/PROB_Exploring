import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import pmath
import itertools


def load_obj_scores(objscores_filename):
    with h5py.File(f'/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/{objscores_filename}.h5', 'r') as file:
        with h5py.File(f'/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/{objscores_filename}_class_name.h5', 'r') as class_name_file:
            print('Number of image samples:', len(file.keys()))
            layers_obj_scores = []
            layers_obj_scores_class_name = []
            
            # Load object data
            for idx, sample_key in enumerate(file.keys()):
                class_names = class_name_file[sample_key][:]
                class_names = [name.decode('utf-8') for name in class_names]
                layers_obj_scores_class_name.extend(class_names)
                layers_obj_scores.append(np.array(file[sample_key]['obj_scores']))

            # Post processing            
            layers_obj_scores = np.concatenate(layers_obj_scores, axis=0)
            print('layers_obj_scores', layers_obj_scores.shape)
            layers_obj_scores_class_name = np.array(layers_obj_scores_class_name)
            print('layers_obj_scores_class_name', layers_obj_scores_class_name.shape)

    return layers_obj_scores, layers_obj_scores_class_name


def histogram_visualization(layers_obj_scores, layers_obj_scores_class_name, suffix_name):
    """
    Visualize histogram of objectness scores separated by class (object vs background).
    
    Args:
        layers_obj_scores: Array of objectness scores, shape (N,)
        layers_obj_scores_class_name: Array of class names, shape (N,)
    """
    # Separate scores by class
    object_mask = layers_obj_scores_class_name != 'background'
    background_mask = layers_obj_scores_class_name == 'background'
    
    obj_scores = layers_obj_scores[object_mask]
    bg_scores = layers_obj_scores[background_mask]
    
    # Calculate statistics
    obj_mean = np.mean(obj_scores)
    obj_std = np.std(obj_scores)
    bg_mean = np.mean(bg_scores)
    bg_std = np.std(bg_scores)
    
    print(f"\nObject scores: count={len(obj_scores)}, mean={obj_mean:.4f}, std={obj_std:.4f}")
    print(f"Background scores: count={len(bg_scores)}, mean={bg_mean:.4f}, std={bg_std:.4f}")
    
    # Create figure with subplots
    fig, axe = plt.subplots(figsize=(10, 6))
    
    axe.hist(obj_scores, bins=50, alpha=0.7, label=f'Object (n={len(obj_scores)})', 
             edgecolor='black', color='blue', density=True)
    axe.hist(bg_scores, bins=50, alpha=0.7, label=f'Background (n={len(bg_scores)})', 
             edgecolor='black', color='red', density=True)
    axe.set_xlabel('Objectness Score', fontsize=12)
    axe.set_ylabel('Frequency', fontsize=12)
    axe.set_title(f'Objectness Score Distribution ({suffix_name})\nObject: μ={obj_mean:.4f}, σ={obj_std:.4f}\nBackground: μ={bg_mean:.4f}, σ={bg_std:.4f}', 
                 fontsize=11)
    axe.legend(fontsize=10)
    axe.grid(True, alpha=0.3)
    
    # Save plot
    save_path = f'/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/obj_scores_histogram_{suffix_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved histogram plot to '{save_path}'")
    plt.close()
    

if __name__ == '__main__':
    # obj_temp = 1.3
    # obj_temp_str = str(obj_temp).replace('.', '_dot_')
    # objscores_filename = f'objscores_V18_Bg_Obj_IoU05_obj_temp_{obj_temp_str}'
    objscores_filename = f'objscores_V10_Bg_Obj_IoU05'
    suffix_name = objscores_filename.replace('objscores_V10', 'PROB').replace('objscores_V18', 'PROB_OBJ_HYP')
    
    layers_obj_scores, layers_obj_scores_class_name = load_obj_scores(objscores_filename)
    histogram_visualization(layers_obj_scores, layers_obj_scores_class_name, suffix_name)

    pass