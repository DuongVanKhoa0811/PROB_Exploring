import os
import cv2
import json
import copy
import torch
import shutil
import numpy as np
from util import box_ops
import matplotlib.cm as cm
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES


def get_class_colors(num_known_classes=20, num_total_classes=81):
    """
    Create color mappings for classes.
    - Known classes (0-19): 20 distinct colors (darker for GT, lighter for predictions)
    - Unknown classes (20-79): dark gray/black (darker for GT, lighter for predictions)
    - 'unknown' class (80): gray (darker for GT, lighter for predictions)
    Returns: (gt_colors_dict, pred_colors_dict) where colors are in BGR format for OpenCV
    """
    # Color intensity multipliers
    GT_INTENSITY = 0.7
    PRED_INTENSITY = 0.9
    
    # Create colormap for 20 known classes
    assert num_known_classes <= 20
    colormap = cm.get_cmap('tab20', 20)
    
    gt_colors = {}
    pred_colors = {}
    
    # Assign colors to known classes (0-19)
    for class_idx in range(num_known_classes):
        # Get color from colormap (RGBA in 0-1 range)
        color_rgba = colormap(class_idx)
        
        # Convert to BGR and scale to 0-255
        # Ground truth: darker colors
        gt_color_bgr = (
            int(color_rgba[2] * 255 * GT_INTENSITY),  # B
            int(color_rgba[1] * 255 * GT_INTENSITY),  # G
            int(color_rgba[0] * 255 * GT_INTENSITY)   # R
        )
        
        # Predictions: lighter colors
        pred_color_bgr = (
            int(color_rgba[2] * 255 * PRED_INTENSITY),  # B
            int(color_rgba[1] * 255 * PRED_INTENSITY),  # G
            int(color_rgba[0] * 255 * PRED_INTENSITY)   # R
        )
        
        gt_colors[class_idx] = gt_color_bgr
        pred_colors[class_idx] = pred_color_bgr
    
    # Assign colors to unknown classes (20-79)
    # Use darker gray/black for GT, lighter gray for predictions
    BASE_BLACK_VALUE = 20
    gt_black = (
        int(BASE_BLACK_VALUE * GT_INTENSITY),
        int(BASE_BLACK_VALUE * GT_INTENSITY),
        int(BASE_BLACK_VALUE * GT_INTENSITY)
    )
    pred_black = (
        int(BASE_BLACK_VALUE * PRED_INTENSITY),
        int(BASE_BLACK_VALUE * PRED_INTENSITY),
        int(BASE_BLACK_VALUE * PRED_INTENSITY)
    )
    
    for class_idx in range(num_known_classes, num_total_classes - 1):  # 20-79
        gt_colors[class_idx] = gt_black
        pred_colors[class_idx] = pred_black
    
    # Assign color to 'unknown' class (80)
    # Use darker gray for GT, lighter gray for predictions
    BASE_GRAY_VALUE = 80
    unknown_class_idx = num_total_classes - 1  # 80
    gt_gray = (
        int(BASE_GRAY_VALUE * GT_INTENSITY),
        int(BASE_GRAY_VALUE * GT_INTENSITY),
        int(BASE_GRAY_VALUE * GT_INTENSITY)
    )
    pred_gray = (
        int(BASE_GRAY_VALUE * PRED_INTENSITY),
        int(BASE_GRAY_VALUE * PRED_INTENSITY),
        int(BASE_GRAY_VALUE * PRED_INTENSITY)
    )
    
    gt_colors[unknown_class_idx] = gt_gray
    pred_colors[unknown_class_idx] = pred_gray
    
    return gt_colors, pred_colors


def read_image_id_to_filename(dataset_name, test_set, data_root):
    from datasets.torchvision_datasets.open_world import OWDetection
    
    def extract_fns(dataset_name, image_set, data_root):
        splits_dir = os.path.join(data_root, 'ImageSets')
        splits_dir = os.path.join(splits_dir, dataset_name)
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        return file_names
    
    file_names = extract_fns(dataset_name, test_set, data_root)
    
    map_image_id_to_filename = {OWDetection.convert_image_id(x, to_integer=True) : x for x in file_names}
    return map_image_id_to_filename


def draw_pred_boxes(results_after_process, targets, dataset_name, test_set, data_root, n_introduce_classes, threshold=0.65, draw_bb_verbose=False):

    save_folder = './trash/bb_img_PROB_OBJ_unknown_only_TOWOD_owod_all_task_test'
    imgs_folder = './data/OWOD/JPEGImages'

    map_image_id_to_filename = read_image_id_to_filename(dataset_name, test_set, data_root)
    dataset_VOC_COCO_CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset_name]
    
    # Get color mappings
    assert n_introduce_classes <= 20
    gt_colors, pred_colors = get_class_colors(num_known_classes=20, num_total_classes=81)
    
    for idx_result in range(len(results_after_process)):
        img_name = map_image_id_to_filename[int(targets[idx_result]['image_id'])] + '.jpg'
        img_path = os.path.join(imgs_folder, img_name)
        np_img = cv2.imread(img_path)

        # Draw the ground truth bounding boxes
        # The targets have been transformed, so we untransform the targets before draw the gt bounding boxes
        np_img_with_gt = np_img.copy()
        targets_copy = copy.deepcopy(targets)
        target_sizes = torch.stack([t["orig_size"] for t in targets_copy], dim=0)        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = box_ops.box_cxcywh_to_xyxy(targets_copy[idx_result]['boxes'][None])
        boxes = boxes * scale_fct[:, None, :]
        boxes = boxes[0]
        for i in range(len(boxes)):
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])
            label = int(targets_copy[idx_result]['labels'][i])
            color = gt_colors.get(label, (0, 0, 0))  # Fallback to black
            # print('eee', label, n_introduce_classes)
            if label >= n_introduce_classes: 
                _text = 'unknown'
            else: 
                _text = dataset_VOC_COCO_CLASS_NAMES[label]
            np_img_with_gt = cv2.rectangle(np_img_with_gt, (x1, y1), 
                                       (x2, y2), color, 2)
            np_img_with_gt = cv2.putText(np_img_with_gt, _text, (x1, y1), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)

        # Draw the predicted bounding boxes
        np_img_with_known_pred = np_img.copy()
        np_img_with_unknown_pred = np_img.copy()
        for i in range(len(results_after_process[idx_result]['boxes'])):
            if results_after_process[idx_result]['scores'][i] < threshold: continue
            label = int(results_after_process[idx_result]['labels'][i])
            x1 = int(results_after_process[idx_result]['boxes'][i][0])
            y1 = int(results_after_process[idx_result]['boxes'][i][1])
            x2 = int(results_after_process[idx_result]['boxes'][i][2])
            y2 = int(results_after_process[idx_result]['boxes'][i][3])
            
            # Get lighter color for predictions
            color = pred_colors.get(label, (0, 0, 0))  # Fallback to black
            
            _text = dataset_VOC_COCO_CLASS_NAMES[label]
            _text += ' ' + str(float(results_after_process[idx_result]['scores'][i]))[:5]
            
            if dataset_VOC_COCO_CLASS_NAMES[label] == 'unknown':
                np_img_with_unknown_pred = cv2.rectangle(np_img_with_unknown_pred, (x1, y1), (x2, y2), (255,0,0), 2)
                # np_img_with_unknown_pred = cv2.putText(np_img_with_unknown_pred, _text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
            else:
                np_img_with_known_pred = cv2.rectangle(np_img_with_known_pred, (x1, y1), (x2, y2), color, 2)
                # np_img_with_known_pred = cv2.putText(np_img_with_known_pred, _text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        # np_img_with_known_pred = cv2.putText(np_img_with_known_pred, 'threshold=' + str(threshold), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        # np_img_with_unknown_pred = cv2.putText(np_img_with_unknown_pred, 'threshold=' + str(threshold), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        
        np_img = np.hstack((np_img_with_gt, np_img_with_known_pred, np_img_with_unknown_pred))
        
        cv2.imwrite(os.path.join(save_folder, img_name.replace('.jpg', '_with_bb.jpg')), np_img)
        if draw_bb_verbose: print('Save draw predicted boxes on image', os.path.join(save_folder, img_name.replace('.jpg', '_with_bb.jpg')))


if __name__ == '__main__':
    
    pass
