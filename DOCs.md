# Streamlit Demo for PROB OWOD

This demo application allows you to visualize unknown object detection results from three PROB models using Streamlit.

## Features

- Upload images and run inference on three models: PROB, PROB_OBJ, PROB_OBJ_HYP
- Filter and display only unknown object detections (ignoring known classes)
- Adjustable Top-K detections
- Side-by-side comparison of results from all three models

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_demo.py
```

2. The app will open in your browser (usually at `http://localhost:8501`)

3. Upload an image:
   - Click "Browse files" or drag and drop an image
   - Supported formats: JPG, JPEG, PNG

5. Run inference:
   - Click "Run Inference" button
   - Results will show:
     - Detection counts for each model
     - Visualizations with bounding boxes drawn on images
     - Detailed detection information



# DEMO

## Steps

### Understand Important Files to Concatenate
Compare Unknown recall@50: PROB, PROB_OBJ, PROB_OBJ_HYP

**PROB**
- main: HEAD
- extract obj features (using IoU): 0d0af39 exp_extract_obj_features: Test #5 - IoU>0.6

**PROB_OBJ**
- exp_obj_train_only: a7cba26 exp_obj_train_only: Test #5 - Eval V16_1

**PROB_OBJ_HYP**
- exp_obj_train_only_test_6: c61f790 exp_obj_train_only: Test #6 - Eval 18_1
- extract obj features (using IoU): a33b38f Update DOCs.md, extract ToPoinCare layer feature

### Targets
We only show the predicted bounding boxes with `class_prediction="unknown"`:
- Collect top-k confidence scores
- IoU with ground truth (gt) unknown > 0.5 (requires the label file)

### Build the Demo Branch
We need to load n models (PROB, PROB_OBJ, PROB_OBJ_HYP) and compare IoU with gt.

**Differences between PROB, PROB_OBJ, PROB_OBJ_HYP:**
- See how PROB, PROB_OBJ, PROB_OBJ_HYP differ in code
- Differences come from the `main_open_world.py` and `prob_deformable_detr.py` files

**Check the difference of extract obj features (using IoU) between PROB and PROB_OBJ_HYP:**
- See how they differ
- Differences come from the `main_open_world.py`, `engine.py` and `prob_deformable_detr.py` files

**Generate the new branch**

## Output

**Unknown classes:** ('truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'unknown')

**Known classes:** ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

...

## Issues

**Why does PROB_OBJ_HYP seem to have lower Unknown Recall@50 than PROB and PROB_OBJ?**
- **Reason:** PROB_OBJ_HYP: top-k scores do not overlap with the unknown objects label. I checked that top-k lowest scores seem to overlap with the unknown objects label.

**Collect samples that include only the unknown class label:**
- `images_containing_exclusively_unknown`: 921 samples (samples_count: 1082)

**Results:**

PROB:
- 2GPU (BS 10) All: 20.45
- 1GPU (BS 1) All: 20.09
- 1GPU (BS 1) images_containing_exclusively_unknown (samples_count: 1082): 6.85

PROB_OBJ_HYP:
- 2GPU (BS 10) All: 22.40
- 1GPU (BS 1) All: 21.88
- 1GPU (BS 1) images_containing_exclusively_unknown: 7.45

**Conclusion:**
- I can now confirm that PROB_OBJ_HYP is better than PROB on Unknown Recall@50 metric
- Visualize such samples, visualize both the gt and predictions