# ------------------------------------------------------------------------
# PROB: Probabilistic Objectness for Open World Object Detection 
# Streamlit Demo Application
# ------------------------------------------------------------------------

import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import argparse
from pathlib import Path
import sys
import os
import shlex

from models import build_model
from main_open_world import get_args_parser
from datasets.coco import make_coco_transforms
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES, OWDetection
from models.prob_extract_obj_features import extract_obj, featureTracker
from util.miscellaneous import read_image_id_to_filename, get_class_colors
from util import box_ops
import xml.etree.ElementTree as ET
import copy

# ============================================================================
# PREDEFINED CHECKPOINT PATHS FOR EACH MODEL
# ============================================================================
# Update these paths to point to your model checkpoints
# Parse arguments for each model from the command-line strings
prob_cmd = "--output_dir exps/MOWODB/PROB_V10/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V10/t1.pth --eval --wandb_project ''"
prob_obj_cmd = "--output_dir exps/MOWODB/PROB_V16_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V16_1/t1.pth --eval --wandb_project ''"
prob_obj_hyp_cmd = "--output_dir exps/MOWODB/PROB_V18_1/eval --dataset TOWOD --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 --train_set owod_t1_train --test_set owod_all_task_test --epochs 191 --lr_drop 35 --model_type prob --obj_loss_coef 8e-4 --obj_temp 1.3 --pretrain exps/MOWODB/PROB_V18_1/t1.pth --eval --wandb_project ''"
# ============================================================================

test_set = 'owod_all_task_test'
dataset_VOC_COCO_CLASS_NAMES = VOC_COCO_CLASS_NAMES['TOWOD']


def assign_args():
    """Parse command-line arguments for each model from the eval commands"""
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])

    # Parse each command string
    args_prob = parser.parse_args(shlex.split(prob_cmd))
    args_prob_obj = parser.parse_args(shlex.split(prob_obj_cmd))
    args_prob_obj_hyp = parser.parse_args(shlex.split(prob_obj_hyp_cmd))
    
    return args_prob, args_prob_obj, args_prob_obj_hyp


@st.cache_resource
def load_models(_args_prob, _args_prob_obj, _args_prob_obj_hyp, device):
    """Load all three models and cache them
    
    Args:
        _args_prob: Model configuration arguments for PROB model
        _args_prob_obj: Model configuration arguments for PROB_OBJ model
        _args_prob_obj_hyp: Model configuration arguments for PROB_OBJ_HYP model
        device: Device to load models on
    """
    device = torch.device(device)
    
    # Assert that all checkpoint paths exist
    assert os.path.exists(_args_prob.pretrain), f"Checkpoint path for 'prob' does not exist: {_args_prob.pretrain}"
    assert os.path.exists(_args_prob_obj.pretrain), f"Checkpoint path for 'prob_obj' does not exist: {_args_prob_obj.pretrain}"
    assert os.path.exists(_args_prob_obj_hyp.pretrain), f"Checkpoint path for 'prob_obj_hyp' does not exist: {_args_prob_obj_hyp.pretrain}"
    
    # Build models
    model_prob, criterion_prob, postprocessors_prob, _ = build_model(_args_prob, mode='prob')
    model_prob_obj, criterion_prob_obj, postprocessors_prob_obj, _ = build_model(_args_prob_obj, mode='prob_obj')
    model_prob_obj_hyp, criterion_prob_obj_hyp, postprocessors_prob_obj_hyp, _ = build_model(_args_prob_obj_hyp, mode='prob_obj_hyp')
    
    # Move to device
    model_prob.to(device)
    model_prob_obj.to(device)
    model_prob_obj_hyp.to(device)
    
    # Set to eval mode
    model_prob.eval()
    model_prob_obj.eval()
    model_prob_obj_hyp.eval()
    
    # Load checkpoints for each model separately
    try:
        checkpoint = torch.load(_args_prob.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        model_prob.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint for prob model: {_args_prob.pretrain}")
    except Exception as e:
        raise e
    
    try:
        checkpoint = torch.load(_args_prob_obj.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        model_prob_obj.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint for prob_obj model: {_args_prob_obj.pretrain}")
    except Exception as e:
        raise e
    
    try:
        checkpoint = torch.load(_args_prob_obj_hyp.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        model_prob_obj_hyp.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint for prob_obj_hyp model: {_args_prob_obj_hyp.pretrain}")
    except Exception as e:
        raise e
    
    return {
        'prob': (model_prob, postprocessors_prob),
        'prob_obj': (model_prob_obj, postprocessors_prob_obj),
        'prob_obj_hyp': (model_prob_obj_hyp, postprocessors_prob_obj_hyp)
    }


def preprocess_image(image):
    """Preprocess image for model inference"""
    # Convert PIL to tensor and normalize
    transform = make_coco_transforms(test_set)
    
    w, h = image.size
    
    # Convert to tensor and normalize (shape: [C, H, W])
    image_tensor = transform[-1](image, None)[0]
    
    # Get original size for postprocessing
    orig_size = torch.tensor([h, w])
    
    return image_tensor, orig_size.unsqueeze(0)


def draw_gt_boxes(image, targets, dataset_name, n_introduce_classes=20):
    """Draw ground truth bounding boxes on image"""
    # Get color mappings
    assert n_introduce_classes <= 20
    gt_colors, _ = get_class_colors(num_known_classes=20, num_total_classes=81)
    dataset_VOC_COCO_CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset_name]
    
    # Convert PIL to numpy array (RGB)
    img_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw the ground truth bounding boxes
    np_img_with_gt = img_bgr.copy()
    targets_copy = copy.deepcopy(targets)
    target_sizes = torch.stack([t["orig_size"] for t in targets_copy], dim=0)        
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = box_ops.box_cxcywh_to_xyxy(targets_copy[0]['boxes'][None])
    boxes = boxes * scale_fct[:, None, :]
    boxes = boxes[0]
    
    for i in range(len(boxes)):
        x1 = int(boxes[i][0])
        y1 = int(boxes[i][1])
        x2 = int(boxes[i][2])
        y2 = int(boxes[i][3])
        label = int(targets_copy[0]['labels'][i])
        color = gt_colors.get(label, (255, 0, 0))  # Fallback to black
        # Hardcode unknown class color
        if label == 80: color = (255, 0, 0)
        
        if label >= n_introduce_classes: 
            label_text = 'unknown'
        else: 
            label_text = dataset_VOC_COCO_CLASS_NAMES[label]
        
        cv2.rectangle(np_img_with_gt, (x1, y1), (x2, y2), color, 2)
        cv2.putText(np_img_with_gt, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(np_img_with_gt, cv2.COLOR_BGR2RGB)
    return img_rgb


def draw_pred_boxes(image, results_after_process, dataset_name, n_introduce_classes=20):
    """Draw filtered prediction boxes on image"""
    # Get color mappings
    assert n_introduce_classes <= 20
    _, pred_colors = get_class_colors(num_known_classes=20, num_total_classes=81)
    dataset_VOC_COCO_CLASS_NAMES = VOC_COCO_CLASS_NAMES[dataset_name]
    
    # Convert PIL to numpy array (RGB)
    img_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw the predicted bounding boxes (filtered by IoU > 0.5)
    np_img_with_pred = img_bgr.copy()
    num_detections = 0
    
    if len(results_after_process) > 0 and len(results_after_process[0]['boxes']) > 0:
        for i in range(len(results_after_process[0]['boxes'])):
            label = int(results_after_process[0]['labels'][i])
            x1 = int(results_after_process[0]['boxes'][i][0])
            y1 = int(results_after_process[0]['boxes'][i][1])
            x2 = int(results_after_process[0]['boxes'][i][2])
            y2 = int(results_after_process[0]['boxes'][i][3])
            
            # Get color for predictions
            color = pred_colors.get(label, (255, 0, 0))  # Fallback to red
            # Hardcode unknown class color
            if label == 80: color = (255, 0, 0)
            
            score = float(results_after_process[0]['scores'][i])
            class_name = dataset_VOC_COCO_CLASS_NAMES[label]
            label_text = f"{class_name} {score:.2f}"
            
            cv2.rectangle(np_img_with_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(np_img_with_pred, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            num_detections += 1
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(np_img_with_pred, cv2.COLOR_BGR2RGB)
    return img_rgb, num_detections


@torch.no_grad()
def run_inference(model, postprocessor, image_tensor, orig_size, device, tracker, invalid_cls_logits, args, targets=None, iou_threshold=0.5):
    """Run inference on a single image and filter by IoU > 0.5"""
    # The model can accept a list of tensors (each [C, H, W]) and will convert to NestedTensor internally
    image_tensor = image_tensor.to(device)
    orig_size = orig_size.to(device)
    # Pass as a list to allow automatic NestedTensor conversion
    outputs = model([image_tensor])
    
    # Use extract_obj to filter predictions with IoU > 0.5
    if targets is not None:
        targets_for_iou = copy.deepcopy(targets)
        obj_features, no_objects, class_name, final_mask = extract_obj(
            outputs, tracker, invalid_cls_logits, args.obj_temp/args.hidden_dim, 
            pred_per_im=100, dataset_name=args.dataset, 
            targets=targets_for_iou, iou_threshold=iou_threshold
        )
    else:
        # If no targets, create a mask that includes all predictions
        final_mask = torch.ones(100, dtype=torch.bool, device=device)
    
    # Postprocess with final_mask
    results = postprocessor['bbox'](outputs, orig_size, final_mask=final_mask)
    return results


def get_target_from_filename(dataset_val, filename):
    """Get target from dataset based on filename"""
    # Extract image name without extension
    img_name = filename.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
    
    # Convert filename to image_id
    from datasets.torchvision_datasets.open_world import OWDetection
    image_id = OWDetection.convert_image_id(img_name, to_integer=True)
    
    # Find index in dataset
    try:
        idx = dataset_val.imgids.index(image_id)
        _, target = dataset_val[idx]
        return target
    except (ValueError, IndexError):
        return None


def main():
    
    st.set_page_config(
        page_title="PROB OWOD Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 PROB: Unknown Object Detection Demo")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Device selection
    device_option = st.sidebar.selectbox(
        "Device",
        ["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
        index=0 if torch.cuda.is_available() else 1
    )
    
    # Parse arguments for each model
    args_prob, args_prob_obj, args_prob_obj_hyp = assign_args()
    
    valid_cls_logits = list(range(args_prob.PREV_INTRODUCED_CLS+args_prob.CUR_INTRODUCED_CLS))
    invalid_cls_logits = list(range(args_prob.PREV_INTRODUCED_CLS+args_prob.CUR_INTRODUCED_CLS, args_prob.num_classes-1))
    valid_cls_names = [dataset_VOC_COCO_CLASS_NAMES[i] for i in valid_cls_logits]
    invalid_cls_names = [dataset_VOC_COCO_CLASS_NAMES[i] for i in invalid_cls_logits]
    
    st.markdown(f""" 
                This demo shows unknown object detection results from three models: PROB, PROB_OBJ, PROB_OBJ_HYP.
                
                **Unknown classes:** {invalid_cls_names}
                
                **Known classes:** {valid_cls_names}
                """)
    
    
    # Set device for all args
    args_prob.device = device_option
    args_prob_obj.device = device_option
    args_prob_obj_hyp.device = device_option
    
    # Set seeds
    torch.manual_seed(args_prob.seed)
    np.random.seed(args_prob.seed)
    
    # Store args in session state
    st.session_state.args_prob = args_prob
    st.session_state.args_prob_obj = args_prob_obj
    st.session_state.args_prob_obj_hyp = args_prob_obj_hyp
    
    # Create dataset_val
    if 'dataset_val' not in st.session_state:
        with st.spinner("Loading dataset..."):
            dataset_val = OWDetection(
                args_prob, 
                args_prob.data_root, 
                image_set=args_prob.test_set, 
                dataset=args_prob.dataset, 
                transforms=make_coco_transforms(args_prob.test_set)
            )
            st.session_state.dataset_val = dataset_val
            st.sidebar.success("✅ Dataset loaded successfully!")
    else:
        dataset_val = st.session_state.dataset_val
        st.sidebar.success("✅ Dataset already loaded!")
    
    # Automatically load models if not already loaded
    if 'models' not in st.session_state or st.session_state.models is None:
        with st.spinner("Loading models..."):
            try:
                models = load_models(args_prob, args_prob_obj, args_prob_obj_hyp, device_option)
                st.session_state.models = models
                
                # Initialize feature trackers for each model
                trackers = {}
                for model_name, (model, _) in models.items():
                    trackers[model_name] = featureTracker(model, variant='DDETR')
                st.session_state.trackers = trackers
                
                st.sidebar.success("✅ Models loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading models: {str(e)}")
                st.session_state.models = None
    else:
        st.sidebar.success("✅ Models already loaded!")
    
    # Check if models are loaded
    if 'models' not in st.session_state or st.session_state.models is None:
        st.error("❌ Failed to load models. Please check the error message above and restart the app.")
        st.stop()
        
    # Image upload
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect unknown objects"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.subheader("Original Image")
        st.image(image, width=300)
        
        # Get target from dataset based on filename
        filename = uploaded_file.name
        target = get_target_from_filename(dataset_val, filename)
        targets = None
        
        if target is None:
            st.warning(f"⚠️ Could not find target for image: {filename}. Please ensure the image exists in the dataset.")
        else:
            # Convert target to list format for processing
            targets = [target]
            # Move target to device
            device = torch.device(device_option)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # Run inference
        if st.button("Run Inference", type="primary"):
            if targets is None:
                st.error("Cannot run inference without target. Please upload a valid image from the dataset.")
            else:
                with st.spinner("Running inference on all three models..."):
                    try:
                        device = torch.device(device_option)
                        models = st.session_state.models
                        trackers = st.session_state.trackers
                        
                        # Preprocess image
                        image_tensor, orig_size = preprocess_image(image)
                        
                        # Get invalid class logits
                        invalid_cls_logits = list(range(
                            st.session_state.args_prob.PREV_INTRODUCED_CLS + st.session_state.args_prob.CUR_INTRODUCED_CLS, 
                            st.session_state.args_prob.num_classes - 1
                        ))
                        
                        # Draw ground truth boxes once (same for all models)
                        gt_image = draw_gt_boxes(
                            image,
                            targets,
                            st.session_state.args_prob.dataset,
                            st.session_state.args_prob.PREV_INTRODUCED_CLS + st.session_state.args_prob.CUR_INTRODUCED_CLS
                        )
                        
                        # Run inference for each model
                        results_dict = {}
                        detection_counts = {}
                        
                        for model_name, (model, postprocessor) in models.items():
                            assert st.session_state.args_prob.num_classes == st.session_state.args_prob_obj.num_classes == st.session_state.args_prob_obj_hyp.num_classes
                            
                            # Get appropriate args for this model
                            if model_name == 'prob':
                                model_args = st.session_state.args_prob
                            elif model_name == 'prob_obj':
                                model_args = st.session_state.args_prob_obj
                            else:  # prob_obj_hyp
                                model_args = st.session_state.args_prob_obj_hyp
                            
                            # Run inference with IoU filtering
                            results = run_inference(
                                model, 
                                postprocessor, 
                                image_tensor, 
                                orig_size, 
                                device,
                                trackers[model_name],
                                invalid_cls_logits,
                                model_args,
                                targets=targets,
                                iou_threshold=0.5
                            )
                            
                            results_dict[model_name] = results
                            
                            # Draw prediction boxes only (GT is drawn separately)
                            pred_image, num_detections = draw_pred_boxes(
                                image, 
                                results, 
                                model_args.dataset,
                                model_args.PREV_INTRODUCED_CLS + model_args.CUR_INTRODUCED_CLS
                            )
                            
                            results_dict[model_name + '_vis'] = pred_image
                            detection_counts[model_name] = num_detections
                        
                        # Store GT image in results
                        results_dict['gt_vis'] = gt_image
                        
                        # Store results in session state
                        st.session_state.results = results_dict
                        st.session_state.detection_counts = detection_counts
                        
                        st.success("Inference completed!")
                        
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display results if available
            if 'results' in st.session_state:
                st.header("Results: Unknown Object Detections (IoU > 0.5)")
                
                # Display detection counts
                st.subheader(f"Filtered Detection Counts (IoU > 0.5 with GT unknown)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    count = st.session_state.detection_counts.get('prob', 0)
                    st.metric("prob", f"{count}")
                with col2:
                    count = st.session_state.detection_counts.get('prob_obj', 0)
                    st.metric("prob_obj", f"{count}")
                with col3:
                    count = st.session_state.detection_counts.get('prob_obj_hyp', 0)
                    st.metric("prob_obj_hyp", f"{count}")
                
                # Display visualizations
                st.subheader("Visualizations (Left: Ground Truth, Right: Filtered Predictions)")
                
                # Create 4 columns: 1 for GT, 3 for model predictions
                col_gt, col1, col2, col3 = st.columns([1, 1, 1, 1])
                
                # Display ground truth in left column
                with col_gt:
                    st.write("**Ground Truth**")
                    if 'gt_vis' in st.session_state.results:
                        st.image(
                            st.session_state.results['gt_vis'],
                            use_container_width=True
                        )
                    else:
                        st.info("No GT")
                
                # Display model predictions in right columns
                model_names = ['prob', 'prob_obj', 'prob_obj_hyp']
                cols = [col1, col2, col3]
                
                for idx, model_name in enumerate(model_names):
                    with cols[idx]:
                        st.write(f"**{model_name}**")
                        if model_name + '_vis' in st.session_state.results:
                            st.image(
                                st.session_state.results[model_name + '_vis'],
                                use_container_width=True
                            )
                        else:
                            st.info("No detections")
                
                # Show detailed results
                with st.expander("Detailed Detection Information"):
                    for model_name in model_names:
                        st.write(f"### {model_name}")
                        if model_name in st.session_state.results:
                            result = st.session_state.results[model_name][0]
                            if len(result['boxes']) > 0:
                                for i in range(len(result['boxes'])):
                                    box = result['boxes'][i].cpu().numpy()
                                    score = result['scores'][i].cpu().item()
                                    st.write(f"  - Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}], Score: {score:.3f}")
                            else:
                                st.write("  - No detections")
                        st.write("")
        

if __name__ == "__main__":
    main()

    pass
