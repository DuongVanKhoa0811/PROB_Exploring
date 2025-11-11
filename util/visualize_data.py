import os
import cv2
import torch
import numpy as np
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES

def visualize_batch(data_loader_train, args, num_images=100):
    """
    Visualize images with bounding boxes from data_loader_train
    
    Args:
        data_loader_train: The training data loader
        args: Arguments object containing dataset info
        num_images: Number of images to save (default 2)
    """
    # ImageNet normalization values used in transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Get class names
    class_names = VOC_COCO_CLASS_NAMES[args.dataset]
    
    data_iter = iter(data_loader_train)
    
    list_labels = []
    
    # Process num_images samples
    for img_idx in range(num_images):
        # Get one batch (should be batch size 1)
        samples, targets = next(data_iter)
        
        # Assert batch size is 1
        assert len(targets) == 1, f"Expected batch size 1, got {len(targets)}"
        
        # Extract images from NestedTensor
        if hasattr(samples, 'tensors'):
            images = samples.tensors
        else:
            images = samples
        
        # Get the single image and target
        img = images[0].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for cv2
        
        target = targets[0]
        boxes = target['boxes'].cpu().numpy()  # cxcywh format (normalized)
        labels = target['labels'].cpu().numpy()
        
        list_labels.extend(labels.tolist())
        
        # Get the actual image size (not the padded size) from mask
        # The mask indicates which regions are padding (True = padding, False = valid image)
        if hasattr(samples, 'mask'):
            mask = samples.mask[0].cpu().numpy()  # True where padding exists
            valid_rows = ~mask.all(axis=1)
            valid_cols = ~mask.all(axis=0)
            actual_h = valid_rows.sum()
            actual_w = valid_cols.sum()
            
            # Crop the image to remove padding
            img = img[:actual_h, :actual_w]
        else:
            # If no mask, use full image dimensions
            actual_h, actual_w = img.shape[:2]
        
        # Draw bounding boxes
        for box, label in zip(boxes, labels):
            # Convert from cxcywh (normalized) to xyxy (pixel coordinates)
            # Boxes are normalized relative to the current image size (including padding if present)
            cx, cy, w, h = box
            
            # Convert normalized cxcywh to normalized xyxy
            x1_norm = cx - 0.5 * w
            y1_norm = cy - 0.5 * h
            x2_norm = cx + 0.5 * w
            y2_norm = cy + 0.5 * h
            
            # Convert to pixel coordinates using actual (non-padded) dimensions
            x1 = int(x1_norm * actual_w)
            y1 = int(y1_norm * actual_h)
            x2 = int(x2_norm * actual_w)
            y2 = int(y2_norm * actual_h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Add label text
            label_text = class_names[label] if label < len(class_names) else f"class_{label}"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 0, 255),
                -1
            )
            
            # Draw text
            cv2.putText(
                img,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Save image
        folder_path = '/home/khoadv/projects/OOD_OD/PROB_Exploring/trash/MOWODB'
        filename = f'train_image_{img_idx}_id_{target["image_id"].item()}_objects_{len(labels)}.png'
        cv2.imwrite(os.path.join(folder_path, filename), img)
        print(f"Saved: {filename}")
    
    import ipdb;ipdb.set_trace()
    print('b', len(set(list_labels)))
    print('c', [class_names[i] for i in set(list_labels)])
    import ipdb;ipdb.set_trace()
    
    print(f"All {num_images} images saved successfully")

