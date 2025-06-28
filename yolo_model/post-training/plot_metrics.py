import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import glob

def plot_training_metrics():
    """
    Plot comprehensive training metrics from YOLOv8 results
    """
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
    print(f"Results CSV: {results_csv}")
    
    # Create results directory
    results_dir = "/home/ubuntu/pharma_system/yolo_model/outputs/plots"
    os.makedirs(results_dir, exist_ok=True)
    
    # Read the results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove any whitespace
    
    print("Available columns:", df.columns.tolist())
    print(f"Data shape: {df.shape}")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a comprehensive figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Pharmaceutical Drug Classification Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    if 'train/loss' in df.columns and 'val/loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(df['epoch'], df['val/loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training vs Validation Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Top-1 and Top-5 Accuracy
    if 'metrics/accuracy_top1' in df.columns:
        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'], label='Top-1 Accuracy', linewidth=2, color='green')
        if 'metrics/accuracy_top5' in df.columns:
            axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top5'], label='Top-5 Accuracy', linewidth=2, color='blue')
        axes[0, 1].set_title('Classification Accuracy', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Learning Rate Schedule
    if 'lr/pg0' in df.columns:
        axes[0, 2].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2, color='red')
        axes[0, 2].set_title('Learning Rate Schedule', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_yscale('log')
    
    # Plot 4: Training Time per Epoch
    if 'train/time' in df.columns:
        axes[1, 0].plot(df['epoch'], df['train/time'], label='Training Time', linewidth=2, color='orange')
        axes[1, 0].set_title('Training Time per Epoch', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: GPU Memory Usage
    if 'train/gpu_mem' in df.columns:
        axes[1, 1].plot(df['epoch'], df['train/gpu_mem'], label='GPU Memory', linewidth=2, color='purple')
        axes[1, 1].set_title('GPU Memory Usage', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Loss Components (if available)
    loss_components = [col for col in df.columns if 'loss' in col.lower() and col != 'val/loss' and col != 'train/loss']
    if loss_components:
        for i, component in enumerate(loss_components[:3]):  # Plot up to 3 components
            axes[1, 2].plot(df['epoch'], df[component], label=component, linewidth=2)
        axes[1, 2].set_title('Loss Components', fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # If no loss components, plot accuracy improvement
        if 'metrics/accuracy_top1' in df.columns:
            accuracy_improvement = df['metrics/accuracy_top1'].diff().rolling(window=5).mean()
            axes[1, 2].plot(df['epoch'], accuracy_improvement, label='Accuracy Improvement (5-epoch avg)', linewidth=2, color='darkgreen')
            axes[1, 2].set_title('Accuracy Improvement Rate', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Accuracy Change')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_path = os.path.join(results_dir, 'training_metrics_comprehensive.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive metrics plot saved to: {plot_path}")
    
    # Create individual plots for key metrics
    
    # Individual plot: Accuracy over time
    if 'metrics/accuracy_top1' in df.columns:
        plt.figure(figsize=(12, 8))
        plt.plot(df['epoch'], df['metrics/accuracy_top1'], linewidth=3, color='green', marker='o', markersize=4)
        if 'metrics/accuracy_top5' in df.columns:
            plt.plot(df['epoch'], df['metrics/accuracy_top5'], linewidth=3, color='blue', marker='s', markersize=4)
        
        plt.title('Pharmaceutical Drug Classification Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(['Top-1 Accuracy', 'Top-5 Accuracy'] if 'metrics/accuracy_top5' in df.columns else ['Top-1 Accuracy'], fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add final accuracy text
        final_acc = df['metrics/accuracy_top1'].iloc[-1]
        plt.text(0.02, 0.98, f'Final Top-1 Accuracy: {final_acc:.3f} ({final_acc*100:.1f}%)', 
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        accuracy_path = os.path.join(results_dir, 'accuracy_plot.png')
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to: {accuracy_path}")
    
    # Individual plot: Loss curves
    if 'train/loss' in df.columns and 'val/loss' in df.columns:
        plt.figure(figsize=(12, 8))
        plt.plot(df['epoch'], df['train/loss'], linewidth=3, color='red', marker='o', markersize=4, label='Training Loss')
        plt.plot(df['epoch'], df['val/loss'], linewidth=3, color='orange', marker='s', markersize=4, label='Validation Loss')
        
        plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add final loss text
        final_train_loss = df['train/loss'].iloc[-1]
        final_val_loss = df['val/loss'].iloc[-1]
        plt.text(0.02, 0.98, f'Final Training Loss: {final_train_loss:.4f}\nFinal Validation Loss: {final_val_loss:.4f}', 
                transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        loss_path = os.path.join(results_dir, 'loss_plot.png')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {loss_path}")
    
    # Copy results.csv to results directory
    results_csv_copy = os.path.join(results_dir, 'training_results.csv')
    df.to_csv(results_csv_copy, index=False)
    print(f"Training results CSV saved to: {results_csv_copy}")
    
    # Generate summary statistics
    summary_stats = {
        'total_epochs': len(df),
        'final_top1_accuracy': df['metrics/accuracy_top1'].iloc[-1] if 'metrics/accuracy_top1' in df.columns else None,
        'final_top5_accuracy': df['metrics/accuracy_top5'].iloc[-1] if 'metrics/accuracy_top5' in df.columns else None,
        'best_top1_accuracy': df['metrics/accuracy_top1'].max() if 'metrics/accuracy_top1' in df.columns else None,
        'best_epoch': df.loc[df['metrics/accuracy_top1'].idxmax(), 'epoch'] if 'metrics/accuracy_top1' in df.columns else None,
        'final_train_loss': df['train/loss'].iloc[-1] if 'train/loss' in df.columns else None,
        'final_val_loss': df['val/loss'].iloc[-1] if 'val/loss' in df.columns else None,
        'min_train_loss': df['train/loss'].min() if 'train/loss' in df.columns else None,
        'min_val_loss': df['val/loss'].min() if 'val/loss' in df.columns else None,
    }
    
    return summary_stats, results_dir

if __name__ == "__main__":
    stats, results_dir = plot_training_metrics()
    if stats:
        print("\nTraining Summary:")
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value}")
        print(f"\nAll plots saved to: {results_dir}")
    else:
        print("Failed to generate plots.") 