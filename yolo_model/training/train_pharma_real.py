from ultralytics import YOLO
import os
import yaml
import pandas as pd
from datetime import datetime
import torch

def train_pharma_classification():
    """
    Train YOLOv8 classification model on pharmaceutical drug dataset
    """
    print("=" * 60)
    print("PHARMACEUTICAL DRUG CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset and model configuration
    data_path = "/home/ubuntu/pharma_system/yolo_model/configs/data.yaml"
    model_name = "yolov8n-cls.pt"  # Nano model for faster training
    
    print(f"Dataset path: {data_path}")
    print(f"Model: {model_name}")
    
    # Load the model
    model = YOLO(model_name)
    
    # Training parameters
    training_params = {
        'data': data_path,
        'epochs': 50,
        'imgsz': 224,
        'batch': 32,
        'device': device,
        'workers': 8,
        'patience': 15,
        'save_period': 10,
        'val': True,
        'plots': True,
        'verbose': True,
        'project': '/home/ubuntu/pharma_system/yolo_model/runs',
        'name': 'pharma_classification',
        'exist_ok': True,
        # Optimization parameters
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.1,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("\nTraining Parameters:")
    for key, value in training_params.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training...")
    print("=" * 60)
    
    # Start training
    results = model.train(**training_params)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    
    # Get the best model path
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    last_model_path = results.save_dir / 'weights' / 'last.pt'
    
    print(f"Best model saved at: {best_model_path}")
    print(f"Last model saved at: {last_model_path}")
    
    # Load best model for evaluation
    best_model = YOLO(best_model_path)
    
    # Validate on test set
    print("\nEvaluating on test set...")
    test_results = best_model.val(data=data_path, split='test')
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Training completed at: {datetime.now()}")
    print(f"Best model: {best_model_path}")
    print(f"Results directory: {results.save_dir}")
    
    # Extract metrics
    if hasattr(test_results, 'top1'):
        print(f"Test Top-1 Accuracy: {test_results.top1:.4f}")
    if hasattr(test_results, 'top5'):
        print(f"Test Top-5 Accuracy: {test_results.top5:.4f}")
    
    return results, best_model_path

if __name__ == "__main__":
    results, model_path = train_pharma_classification() 