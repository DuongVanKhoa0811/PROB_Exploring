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

## Troubleshooting

- **No detections**

