import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    list_obj_prob_PROB_OBJ_obj_temp_1_dot_3 = torch.load('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/list_obj_prob_PROB_OBJ_obj_temp_1_dot_3.pth')
    list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3 = torch.load('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3.pth')
    list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3 = torch.load('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3.pth')
    list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3 = torch.load('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3.pth')
    
    list_obj_prob_PROB_OBJ_obj_temp_1_dot_3 = list_obj_prob_PROB_OBJ_obj_temp_1_dot_3.flatten().cpu()
    list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3 = list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3.flatten().cpu()
    list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3 = list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3.flatten().cpu()
    list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3 = list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3.flatten().cpu()
    
    print('list_obj_prob_PROB_OBJ_obj_temp_1_dot_3', list_obj_prob_PROB_OBJ_obj_temp_1_dot_3.shape)
    print('list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3', list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3.shape)
    print('list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3', list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3.shape)
    print('list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3', list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3.shape)
    
    # Calculate statistics
    mean_1 = list_obj_prob_PROB_OBJ_obj_temp_1_dot_3.mean()
    std_1 = list_obj_prob_PROB_OBJ_obj_temp_1_dot_3.std()
    
    mean_2 = list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3.mean()
    std_2 = list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3.std()
    
    mean_3 = list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3.mean()
    std_3 = list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3.std()
    
    mean_4 = list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3.mean()
    std_4 = list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3.std()
    
    # Use separate subplots for better comparison
    title_fontsize = 19
    label_fontsize = 16
    tick_fontsize = 16
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    axes[0,0].hist(list_obj_prob_PROB_OBJ_obj_temp_1_dot_3, bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title(f'PROB_OBJ_obj_temp_default\n(Mean: {mean_1:.4f}, Std: {std_1:.4f})', fontsize=title_fontsize)
    axes[0,0].set_xlabel('Objectness Score', fontsize=label_fontsize)
    axes[0,0].set_ylabel('Frequency', fontsize=label_fontsize)
    axes[0,0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].hist(list_obj_prob_PROB_OBJ_HYP_obj_temp_1_dot_3, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title(f'PROB_OBJ_HYP_obj_temp_default\n(Mean: {mean_2:.4f}, Std: {std_2:.4f})', fontsize=title_fontsize)
    axes[0,1].set_xlabel('Objectness Score', fontsize=label_fontsize)
    axes[0,1].set_ylabel('Frequency', fontsize=label_fontsize)
    axes[0,1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].hist(list_obj_prob_PROB_OBJ_HYP_obj_temp_105_dot_3, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1,0].set_title(f'PROB_OBJ_HYP_81_times_obj_temp_default\n(Mean: {mean_3:.4f}, Std: {std_3:.4f})', fontsize=title_fontsize)
    axes[1,0].set_xlabel('Objectness Score', fontsize=label_fontsize)
    axes[1,0].set_ylabel('Frequency', fontsize=label_fontsize)
    axes[1,0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].hist(list_obj_prob_PROB_OBJ_HYP_obj_temp_157_dot_3, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1,1].set_title(f'PROB_OBJ_HYP_121_times_obj_temp_default\n(Mean: {mean_4:.4f}, Std: {std_4:.4f})', fontsize=title_fontsize)
    axes[1,1].set_xlabel('Objectness Score', fontsize=label_fontsize)
    axes[1,1].set_ylabel('Frequency', fontsize=label_fontsize)
    axes[1,1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Objectness Score Distributions (Separate Views)', fontsize=title_fontsize, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/obj_score_visualization.png', dpi=150)
    plt.close()