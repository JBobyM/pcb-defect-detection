#!/usr/bin/env python3
# PCB Defect Detection Training Script
# -----------------------------------
# For use with DsPCBSD+ dataset
# Optimized for dual RTX 3090 setup

import os
import torch
import yaml
from ultralytics import YOLO
from pathlib import Path

# Configuration
PROJECT_ROOT = "/work/boby/PCB_project"
DATA_ROOT = "/work/boby/PCB_project/data"
DATASET_CONFIG = os.path.join(PROJECT_ROOT, "dataset.yaml")
BATCH_SIZE_PER_GPU = 16  # 32 total with 2 GPUs
NUM_WORKERS = 16  # Threadripper has plenty of cores
IMAGE_SIZE = 640
EPOCHS = 100
MODEL_TYPE = "yolov8m.pt"  # Medium size model for good speed/accuracy balance

# Create project directories
os.makedirs(os.path.join(PROJECT_ROOT, "runs"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

# Create dataset configuration file
def create_dataset_config():
    config = {
        'path': DATA_ROOT + "/DsPCBSD+",  # dataset root dir
        'train': "Data_YOLO/images/train", 
        'val': "Data_YOLO/images/val",
        
        'names': {
            0: 'SH',   # Short Circuit
            1: 'SP',   # Spur
            2: 'SC',   # Spurious Copper
            3: 'OP',   # Open Circuit
            4: 'MB',   # Mousebite
            5: 'HB',   # Hole Break
            6: 'CS',   # Conductor Scratch
            7: 'CFO',  # Copper Foreign Object
            8: 'BMFO'  # Base Material Foreign Object
        }
    }
    
    with open(DATASET_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created dataset config at {DATASET_CONFIG}")
    return DATASET_CONFIG

def train_model():
    # Ensure dataset config exists
    if not os.path.exists(DATASET_CONFIG):
        create_dataset_config()
    
    # Set device configuration
    # YOLOv8 automatically handles multi-GPU with DDP
    device = '0,1'  # Use both GPUs
    
    # Create model with pre-trained weights
    model = YOLO(MODEL_TYPE)
    
    # Start training
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE_PER_GPU,
        imgsz=IMAGE_SIZE,
        workers=NUM_WORKERS,
        device=device,
        
        # Optimization settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss settings
        cls=0.5,      # Class loss gain
        box=7.5,      # Box loss gain
        
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=3.0,  # Rotation - keep small for PCB defects
        translate=0.1,
        scale=0.4,
        shear=0.0,    # No shear for PCB images
        perspective=0.0,  # No perspective change for PCB images
        flipud=0.0,   # No vertical flip for PCB orientation
        fliplr=0.5,   # Horizontal flip is acceptable
        mosaic=1.0,   # Use mosaic augmentation
        mixup=0.1,    # Light mixup for PCB defects
        copy_paste=0.1,  # Some copy-paste for rare defects
        
        # Saving and logging
        project=os.path.join(PROJECT_ROOT, "runs"),
        name='pcb_defect_detection',
        save=True,
        save_period=5,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42
    )
    
    # Save the final model
    model_path = os.path.join(PROJECT_ROOT, "models", "pcb_detector.pt")
    model.model.save(model_path)
    
    return results, model

def evaluate_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "runs/pcb_defect_detection/weights/best.pt")
    
    model = YOLO(model_path)
    
    # Run validation on the test set
    results = model.val(
        data=DATASET_CONFIG,
        batch=BATCH_SIZE_PER_GPU,
        imgsz=IMAGE_SIZE,
        device=0,  # Use first GPU for validation
        verbose=True,
        conf=0.25,
        iou=0.7,
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    # Print metrics by class
    metrics = results.box
    print("\nPer-Class Performance:")
    print(f"{'Class':<10}{'Precision':<10}{'Recall':<10}{'mAP50':<10}")
    print("-" * 40)
    
    for i, class_name in enumerate(results.names):
        try:
            cls_precision = metrics.mp_per_class[i] if hasattr(metrics, 'mp_per_class') else 0
            cls_recall = metrics.mr_per_class[i] if hasattr(metrics, 'mr_per_class') else 0
            cls_map50 = metrics.map50_per_class[i] if hasattr(metrics, 'map50_per_class') else 0
            
            print(f"{class_name:<10}{cls_precision:.4f}    {cls_recall:.4f}    {cls_map50:.4f}")
        except:
            print(f"{class_name:<10}No data")
    
    print("\nOverall Performance:")
    print(f"mAP@0.5: {metrics.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.map:.4f}")
    print(f"Precision: {metrics.mp:.4f}")
    print(f"Recall: {metrics.mr:.4f}")
    
    return results

if __name__ == "__main__":
    print("PCB Defect Detection - Training Script")
    print(f"Using PyTorch {torch.__version__} with {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Check CUDA devices
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Train the model
    results, model = train_model()
    
    # Evaluate the model
    evaluate_model()
    
    print("Training and evaluation complete!")