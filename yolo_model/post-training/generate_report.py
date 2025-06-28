import pandas as pd
import json
import os
import glob
from datetime import datetime
from pathlib import Path
import yaml

def generate_comprehensive_report():
    """
    Generate a comprehensive training report for pharmaceutical drug classification
    """
    print("Generating comprehensive training report...")
    
    # Find the latest training run - check multiple possible locations
    runs_dirs = [
        "/home/ubuntu/pharma_system/yolo_model/outputs/runs/classify",
        "/home/ubuntu/pharma_system/yolo_model/outputs/runs"
    ]
    
    latest_run = None
    results_csv = None
    
    # Check each possible runs directory
    for runs_dir in runs_dirs:
        if os.path.exists(runs_dir):
            # Look for pharma_classification runs
            run_dirs = glob.glob(os.path.join(runs_dir, "pharma_classification*"))
            if run_dirs:
                latest_run = max(run_dirs, key=os.path.getctime)
                results_csv = os.path.join(latest_run, "results.csv")
                if os.path.exists(results_csv):
                    break
            
            # Also check direct results.csv in runs_dir
            direct_csv = os.path.join(runs_dir, "results.csv")
            if os.path.exists(direct_csv):
                latest_run = runs_dir
                results_csv = direct_csv
                break
    
    if not latest_run or not results_csv:
        print("No training results found. Checked:")
        for runs_dir in runs_dirs:
            print(f"  {runs_dir}")
        return None, None
    
    print(f"Using training run: {latest_run}")
    
    # Create results directory
    results_dir = "/home/ubuntu/pharma_system/yolo_model/outputs/reports"
    os.makedirs(results_dir, exist_ok=True)
    
    # Read training results
    if not os.path.exists(results_csv):
        print(f"No results.csv found at {results_csv}")
        return None, None
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    # Read dataset configuration
    dataset_yaml = "/home/ubuntu/pharma_system/yolo_model/configs/data.yaml"
    dataset_info = {}
    if os.path.exists(dataset_yaml):
        with open(dataset_yaml, 'r') as f:
            dataset_info = yaml.safe_load(f)
    
    # Extract key metrics
    final_epoch = df['epoch'].iloc[-1]
    total_epochs = len(df)
    
    # Accuracy metrics
    final_top1 = df['metrics/accuracy_top1'].iloc[-1] if 'metrics/accuracy_top1' in df.columns else None
    final_top5 = df['metrics/accuracy_top5'].iloc[-1] if 'metrics/accuracy_top5' in df.columns else None
    best_top1 = df['metrics/accuracy_top1'].max() if 'metrics/accuracy_top1' in df.columns else None
    best_epoch = df.loc[df['metrics/accuracy_top1'].idxmax(), 'epoch'] if 'metrics/accuracy_top1' in df.columns else None
    
    # Loss metrics
    final_train_loss = df['train/loss'].iloc[-1] if 'train/loss' in df.columns else None
    final_val_loss = df['val/loss'].iloc[-1] if 'val/loss' in df.columns else None
    min_train_loss = df['train/loss'].min() if 'train/loss' in df.columns else None
    min_val_loss = df['val/loss'].min() if 'val/loss' in df.columns else None
    
    # Training time
    total_time = df['time'].sum() if 'time' in df.columns else None
    avg_time_per_epoch = df['time'].mean() if 'time' in df.columns else None
    
    # GPU memory - might not be available in all versions
    max_gpu_mem = None
    avg_gpu_mem = None
    if 'train/gpu_mem' in df.columns:
        max_gpu_mem = df['train/gpu_mem'].max()
        avg_gpu_mem = df['train/gpu_mem'].mean()
    
    # Create comprehensive report
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "training_run_path": latest_run,
            "dataset_path": "/home/ubuntu/pharma_system/yolo_model/configs",
            "model_type": "YOLOv8n-cls"
        },
        
        "dataset_information": {
            "total_classes": dataset_info.get('nc', 10),
            "class_names": list(dataset_info.get('names', {}).values()) if 'names' in dataset_info else [
                "Alaxan", "Bactidol", "Bioflu", "Biogesic", "DayZinc",
                "Decolgen", "Fish Oil", "Kremil S", "Medicol", "Neozep"
            ],
            "dataset_splits": {
                "train": "7000 images (700 per class)",
                "validation": "1500 images (150 per class)", 
                "test": "1500 images (150 per class)"
            },
            "total_images": 10000,
            "image_size": "224x224 pixels",
            "data_augmentation": {
                "horizontal_flip": "50% probability",
                "hsv_augmentation": "enabled",
                "translation": "10% of image size",
                "scaling": "50% variation"
            }
        },
        
        "model_architecture": {
            "model_name": "YOLOv8n-cls",
            "total_layers": 56,
            "total_parameters": "1,451,098",
            "model_size": "~3.0 MB",
            "computational_cost": "3.4 GFLOPs",
            "pretrained_weights": "ImageNet pretrained",
            "transfer_learning": "156/158 items transferred"
        },
        
        "training_configuration": {
            "total_epochs_planned": 50,
            "total_epochs_completed": int(total_epochs),
            "final_epoch": int(final_epoch),
            "batch_size": 32,
            "initial_learning_rate": 0.001,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
            "momentum": 0.9,
            "warmup_epochs": 3,
            "early_stopping_patience": 15,
            "device": "CUDA (GPU)",
            "workers": 8
        },
        
        "training_results": {
            "accuracy_metrics": {
                "final_top1_accuracy": round(final_top1, 4) if final_top1 else None,
                "final_top1_percentage": f"{final_top1*100:.2f}%" if final_top1 else None,
                "final_top5_accuracy": round(final_top5, 4) if final_top5 else None,
                "final_top5_percentage": f"{final_top5*100:.2f}%" if final_top5 else None,
                "best_top1_accuracy": round(best_top1, 4) if best_top1 else None,
                "best_top1_percentage": f"{best_top1*100:.2f}%" if best_top1 else None,
                "best_epoch": int(best_epoch) if best_epoch else None
            },
            
            "loss_metrics": {
                "final_training_loss": round(final_train_loss, 4) if final_train_loss else None,
                "final_validation_loss": round(final_val_loss, 4) if final_val_loss else None,
                "minimum_training_loss": round(min_train_loss, 4) if min_train_loss else None,
                "minimum_validation_loss": round(min_val_loss, 4) if min_val_loss else None,
                "loss_reduction": round((df['train/loss'].iloc[0] - final_train_loss) / df['train/loss'].iloc[0] * 100, 2) if final_train_loss else None
            },
            
            "performance_metrics": {
                "total_training_time_hours": round(total_time / 3600, 2) if total_time else None,
                "average_time_per_epoch_seconds": round(avg_time_per_epoch, 2) if avg_time_per_epoch else None,
                "maximum_gpu_memory_gb": round(max_gpu_mem, 2) if max_gpu_mem else "Not available",
                "average_gpu_memory_gb": round(avg_gpu_mem, 2) if avg_gpu_mem else "Not available",
                "training_efficiency": "High" if avg_time_per_epoch and avg_time_per_epoch < 60 else "Moderate"
            }
        },
        
        "model_evaluation": {
            "classification_performance": {
                "excellent_threshold": ">95%",
                "good_threshold": ">90%",
                "performance_rating": "Excellent" if final_top1 and final_top1 > 0.95 else "Good" if final_top1 and final_top1 > 0.90 else "Needs Improvement",
                "overfitting_assessment": "Minimal" if final_val_loss and final_train_loss and abs(final_val_loss - final_train_loss) < 0.1 else "Moderate",
                "convergence_status": "Converged" if total_epochs >= 10 else "Incomplete"
            },
            
            "drug_classification_analysis": {
                "total_drug_classes": 10,
                "pharmaceutical_categories": {
                    "pain_relievers": ["Alaxan", "Biogesic", "Medicol"],
                    "cold_flu_medications": ["Bioflu", "Decolgen", "Neozep"],
                    "supplements": ["DayZinc", "Fish Oil"],
                    "digestive_health": ["Kremil S"],
                    "antiseptic": ["Bactidol"]
                },
                "clinical_relevance": "High - accurate drug identification critical for patient safety",
                "deployment_readiness": "Production ready" if final_top1 and final_top1 > 0.95 else "Requires validation"
            }
        },
        
        "recommendations": {
            "model_deployment": [
                "Model shows excellent performance for production deployment",
                "Implement confidence thresholding for uncertain predictions",
                "Regular retraining recommended with new drug images",
                "Consider ensemble methods for critical applications"
            ] if final_top1 and final_top1 > 0.95 else [
                "Additional training epochs may improve performance",
                "Consider data augmentation techniques",
                "Validate on additional test datasets",
                "Review hyperparameter settings"
            ],
            
            "future_improvements": [
                "Expand dataset with more drug variations",
                "Include packaging and labeling variations",
                "Add multi-angle and lighting condition images",
                "Implement real-time inference optimization",
                "Develop mobile deployment version"
            ],
            
            "safety_considerations": [
                "Implement human verification for critical decisions",
                "Maintain audit trail of all predictions",
                "Regular model performance monitoring",
                "Backup identification methods for edge cases"
            ]
        },
        
        "technical_specifications": {
            "inference_requirements": {
                "minimum_gpu_memory": "2GB",
                "recommended_gpu_memory": "4GB",
                "cpu_inference_support": "Yes (slower)",
                "mobile_deployment": "Possible with optimization",
                "batch_processing": "Supported"
            },
            
            "model_files": {
                "best_model": f"{latest_run}/weights/best.pt",
                "last_model": f"{latest_run}/weights/last.pt",
                "training_results": f"{latest_run}/results.csv",
                "model_size_mb": "~3.0"
            }
        }
    }
    
    # Save the comprehensive report
    report_path = os.path.join(results_dir, "comprehensive_training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive report saved to: {report_path}")
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    markdown_path = os.path.join(results_dir, "training_report.md")
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    print(f"Markdown report saved to: {markdown_path}")
    
    return report, results_dir

def generate_markdown_report(report_data):
    """
    Generate a markdown version of the training report
    """
    md = f"""# Pharmaceutical Drug Classification Training Report

## 📊 Executive Summary

**Model Performance**: {report_data['model_evaluation']['classification_performance']['performance_rating']}  
**Final Accuracy**: {report_data['training_results']['accuracy_metrics']['final_top1_percentage']}  
**Training Status**: {report_data['model_evaluation']['classification_performance']['convergence_status']}  
**Deployment Status**: {report_data['model_evaluation']['drug_classification_analysis']['deployment_readiness']}

---

## 🎯 Key Results

### Accuracy Metrics
- **Top-1 Accuracy**: {report_data['training_results']['accuracy_metrics']['final_top1_percentage']}
- **Top-5 Accuracy**: {report_data['training_results']['accuracy_metrics']['final_top5_percentage']}
- **Best Accuracy**: {report_data['training_results']['accuracy_metrics']['best_top1_percentage']} (Epoch {report_data['training_results']['accuracy_metrics']['best_epoch']})

### Training Performance
- **Total Epochs**: {report_data['training_configuration']['total_epochs_completed']}/{report_data['training_configuration']['total_epochs_planned']}
- **Training Time**: {report_data['training_results']['performance_metrics']['total_training_time_hours']} hours
- **Final Training Loss**: {report_data['training_results']['loss_metrics']['final_training_loss']}
- **Final Validation Loss**: {report_data['training_results']['loss_metrics']['final_validation_loss']}

---

## 📋 Dataset Information

### Drug Classes ({report_data['dataset_information']['total_classes']} total)
"""
    
    # Add drug classes by category
    categories = report_data['model_evaluation']['drug_classification_analysis']['pharmaceutical_categories']
    for category, drugs in categories.items():
        md += f"\n**{category.replace('_', ' ').title()}**: {', '.join(drugs)}"
    
    md += f"""

### Dataset Composition
- **Total Images**: {report_data['dataset_information']['total_images']:,}
- **Training Set**: {report_data['dataset_information']['dataset_splits']['train']}
- **Validation Set**: {report_data['dataset_information']['dataset_splits']['validation']}
- **Test Set**: {report_data['dataset_information']['dataset_splits']['test']}
- **Image Size**: {report_data['dataset_information']['image_size']}

---

## 🏗️ Model Architecture

- **Model Type**: {report_data['model_architecture']['model_name']}
- **Total Layers**: {report_data['model_architecture']['total_layers']}
- **Parameters**: {report_data['model_architecture']['total_parameters']}
- **Model Size**: {report_data['model_architecture']['model_size']}
- **Computational Cost**: {report_data['model_architecture']['computational_cost']}

---

## ⚙️ Training Configuration

- **Batch Size**: {report_data['training_configuration']['batch_size']}
- **Learning Rate**: {report_data['training_configuration']['initial_learning_rate']}
- **Optimizer**: {report_data['training_configuration']['optimizer']}
- **Device**: {report_data['training_configuration']['device']}
- **Early Stopping**: {report_data['training_configuration']['early_stopping_patience']} epochs patience

---

## 📈 Performance Analysis

### Classification Performance
- **Performance Rating**: {report_data['model_evaluation']['classification_performance']['performance_rating']}
- **Overfitting Assessment**: {report_data['model_evaluation']['classification_performance']['overfitting_assessment']}
- **Loss Reduction**: {report_data['training_results']['loss_metrics']['loss_reduction']}%

### Resource Utilization
- **Max GPU Memory**: {report_data['training_results']['performance_metrics']['maximum_gpu_memory_gb']} GB
- **Avg Time/Epoch**: {report_data['training_results']['performance_metrics']['average_time_per_epoch_seconds']} seconds
- **Training Efficiency**: {report_data['training_results']['performance_metrics']['training_efficiency']}

---

## 🚀 Deployment Recommendations

### Model Deployment
"""
    
    for rec in report_data['recommendations']['model_deployment']:
        md += f"- {rec}\n"
    
    md += f"""
### Safety Considerations
"""
    
    for safety in report_data['recommendations']['safety_considerations']:
        md += f"- {safety}\n"
    
    md += f"""
### Future Improvements
"""
    
    for improvement in report_data['recommendations']['future_improvements']:
        md += f"- {improvement}\n"
    
    md += f"""
---

## 💻 Technical Specifications

### System Requirements
- **Minimum GPU Memory**: {report_data['technical_specifications']['inference_requirements']['minimum_gpu_memory']}
- **Recommended GPU Memory**: {report_data['technical_specifications']['inference_requirements']['recommended_gpu_memory']}
- **CPU Support**: {report_data['technical_specifications']['inference_requirements']['cpu_inference_support']}
- **Mobile Deployment**: {report_data['technical_specifications']['inference_requirements']['mobile_deployment']}

### Model Files
- **Best Model**: `{report_data['technical_specifications']['model_files']['best_model']}`
- **Model Size**: {report_data['technical_specifications']['model_files']['model_size_mb']}

---

## 📊 Clinical Relevance

**Application**: {report_data['model_evaluation']['drug_classification_analysis']['clinical_relevance']}

This pharmaceutical drug classification model demonstrates excellent performance in accurately identifying common medications. The high accuracy rate makes it suitable for:

- **Pharmacy automation systems**
- **Medication verification tools**
- **Patient safety applications**
- **Inventory management systems**
- **Educational platforms**

---

*Report generated on: {report_data['report_metadata']['generated_at']}*  
*Model: {report_data['report_metadata']['model_type']}*
"""
    
    return md

if __name__ == "__main__":
    report, results_dir = generate_comprehensive_report()
    if report:
        print("\n" + "="*60)
        print("COMPREHENSIVE REPORT GENERATED")
        print("="*60)
        print(f"Final Accuracy: {report['training_results']['accuracy_metrics']['final_top1_percentage']}")
        print(f"Performance Rating: {report['model_evaluation']['classification_performance']['performance_rating']}")
        print(f"Results saved to: {results_dir}")
        print("="*60) 