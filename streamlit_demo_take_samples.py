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
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES

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


def filter_unknown_predictions(results, num_classes, top_k_detections=100):
    """Filter predictions to only show unknown bounding boxes, sorted by score (top-k)"""
    unknown_class_idx = num_classes - 1
    
    filtered_results = []
    for result in results:
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']
        
        assert torch.all(scores[:-1] >= scores[1:]), "Tensor values are not in descending order"
        
        # Filter for unknown class predictions
        unknown_mask = labels == unknown_class_idx
        
        filtered_scores = scores[unknown_mask]
        filtered_labels = labels[unknown_mask]
        filtered_boxes = boxes[unknown_mask]
        
        # Sort by score (descending) and take top-k
        if len(filtered_scores) > 0:
            top_k = min(top_k_detections, len(filtered_scores))
            
            filtered_result = {
                'scores': filtered_scores[:top_k],
                'labels': filtered_labels[:top_k],
                'boxes': filtered_boxes[:top_k]
            }
        else:
            filtered_result = {
                'scores': torch.tensor([]),
                'labels': torch.tensor([]),
                'boxes': torch.tensor([])
            }
        
        filtered_results.append(filtered_result)
    
    return filtered_results


def draw_pred_boxes(image, results_after_process, model_name, n_introduce_classes=20):

    # Get color mappings
    assert n_introduce_classes <= 20
    
    # Convert PIL to numpy array (RGB)
    img_np = np.array(image)
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw the predicted bounding boxes
    for i in range(len(results_after_process[0]['boxes'])):
        label = int(results_after_process[0]['labels'][i])
        x1 = int(results_after_process[0]['boxes'][i][0])
        y1 = int(results_after_process[0]['boxes'][i][1])
        x2 = int(results_after_process[0]['boxes'][i][2])
        y2 = int(results_after_process[0]['boxes'][i][3])
        
        # Get lighter color for predictions
        color = (255, 0, 0)
        
        # _text = str(float(results_after_process[0]['scores'][i]))[:5]
        
        img_bgr = cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        # img_bgr = cv2.putText(img_bgr, _text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        
    num_detections = len(results_after_process[0]['boxes'])
    
    # # Add model name and count
    # info_text = f"{model_name}: {num_detections} unknown detections"
    # cv2.putText(img_bgr, info_text, (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, num_detections


@torch.no_grad()
def run_inference(model, postprocessor, image_tensor, orig_size, device):
    """Run inference on a single image"""
    # The model can accept a list of tensors (each [C, H, W]) and will convert to NestedTensor internally
    image_tensor = image_tensor.to(device)
    orig_size = orig_size.to(device)
    # Pass as a list to allow automatic NestedTensor conversion
    outputs = model([image_tensor])
    results = postprocessor['bbox'](outputs, orig_size)
    return results


def main():
    
    st.set_page_config(
        page_title="PROB OWOD Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 PROB: Unknown Object Detection Demo")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Score threshold
    top_k_detections = st.sidebar.slider(
        "Top-K detections",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )
    
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
    
    # Automatically load models if not already loaded
    if 'models' not in st.session_state or st.session_state.models is None:
        with st.spinner("Loading models..."):
            try:
                models = load_models(args_prob, args_prob_obj, args_prob_obj_hyp, device_option)
                st.session_state.models = models
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
    
    # if uploaded_file is not None: # eeee
    import time
    uploaded_files = [os.path.join('./trash/unknown_only_TOWOD_owod_all_task_test', f) for f in os.listdir('./trash/unknown_only_TOWOD_owod_all_task_test')]
    for uploaded_file in uploaded_files:
        time.sleep(2)
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        st.subheader("Original Image")
        st.image(image, width=300)
        
        # Run inference
        # if st.button("Run Inference", type="primary"):
        if True:
            with st.spinner("Running inference on all three models..."):
                try:
                    device = torch.device(device_option)
                    models = st.session_state.models
                    
                    # Preprocess image
                    image_tensor, orig_size = preprocess_image(image)
                    
                    # Run inference for each model
                    results_dict = {}
                    detection_counts = {}
                    
                    for model_name, (model, postprocessor) in models.items():
                        assert st.session_state.args_prob.num_classes == st.session_state.args_prob_obj.num_classes == st.session_state.args_prob_obj_hyp.num_classes
                        
                        results = run_inference(model, postprocessor, image_tensor, orig_size, device)
                        
                        # Filter for unknown predictions
                        filtered_results = filter_unknown_predictions(
                            results, 
                            st.session_state.args_prob.num_classes, 
                            top_k_detections,
                        )
                        
                        results_dict[model_name] = filtered_results
                        
                        # Draw boxes
                        vis_image, num_detections = draw_pred_boxes(
                            image, 
                            filtered_results, 
                            model_name
                        )
                        
                        results_dict[model_name + '_vis'] = vis_image
                        detection_counts[model_name] = num_detections
                    
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
                st.header("Results: Unknown Object Detections")
                
                # Display detection counts
                st.subheader(f"Unknown Detection Counts (Top-K: {top_k_detections})")
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
                st.subheader("Visualizations")
                cols = st.columns(3)
                
                model_names = ['prob', 'prob_obj', 'prob_obj_hyp']
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

    
    # image = Image.open('/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/000001.jpg').convert('RGB')
    # image_tensor, orig_size = preprocess_image(image)


    # parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    # args = parser.parse_args()

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)

    # args.device = 'cuda'

    # models = load_models(args)
    
    # for model_name, (model, postprocessor) in models.items():
    #     results = run_inference(model, postprocessor, image_tensor, orig_size, 'cuda')

    #     # Filter for unknown predictions
    #     filtered_results = filter_unknown_predictions(
    #         results, 
    #         args.num_classes, 
    #         10,
    #     )
    
    
    pass
