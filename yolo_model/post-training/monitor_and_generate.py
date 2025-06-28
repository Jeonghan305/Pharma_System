#!/usr/bin/env python3

import time
import subprocess
import os
import sys
from pathlib import Path

def check_training_status():
    """Check if training is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'train_pharma_real.py' in result.stdout
    except:
        return False

def wait_for_training_completion():
    """Wait for training to complete"""
    print("🔍 Monitoring training progress...")
    print("=" * 60)
    
    last_check_time = time.time()
    check_interval = 30  # Check every 30 seconds
    
    while True:
        if check_training_status():
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                # Show latest progress
                try:
                    with open('/home/ubuntu/pharma_system/yolo_model/training_real.log', 'r') as f:
                        lines = f.readlines()
                        # Get last few lines that contain epoch info
                        for line in reversed(lines[-20:]):
                            if 'Epoch' in line and '/' in line:
                                print(f"📊 Current: {line.strip()}")
                                break
                except:
                    pass
                
                last_check_time = current_time
            
            time.sleep(10)  # Wait 10 seconds before next check
        else:
            print("\n✅ Training completed!")
            break
    
    # Wait a bit more to ensure all files are written
    print("⏳ Waiting for files to be finalized...")
    time.sleep(10)

def generate_results():
    """Generate plots and comprehensive report"""
    print("\n🎨 Generating training plots...")
    print("=" * 60)
    
    # Generate plots
    try:
        import plot_metrics
        stats, results_dir = plot_metrics.plot_training_metrics()
        print("✅ Training plots generated successfully!")
        
        # Print key statistics
        if stats:
            print("\n📊 Key Training Statistics:")
            for key, value in stats.items():
                if value is not None:
                    print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return False
    
    print("\n📝 Generating comprehensive report...")
    print("=" * 60)
    
    # Generate comprehensive report
    try:
        import generate_report
        report, results_dir = generate_report.generate_comprehensive_report()
        
        if report:
            print("✅ Comprehensive report generated successfully!")
            
            # Print executive summary
            print("\n🎯 EXECUTIVE SUMMARY")
            print("=" * 60)
            accuracy = report['training_results']['accuracy_metrics']['final_top1_percentage']
            rating = report['model_evaluation']['classification_performance']['performance_rating']
            deployment = report['model_evaluation']['drug_classification_analysis']['deployment_readiness']
            
            print(f"🎯 Final Accuracy: {accuracy}")
            print(f"📈 Performance Rating: {rating}")
            print(f"🚀 Deployment Status: {deployment}")
            print(f"📁 Results Directory: {results_dir}")
            
            return True
        else:
            print("❌ Failed to generate comprehensive report")
            return False
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False

def main():
    """Main monitoring and generation function"""
    print("🚀 PHARMACEUTICAL DRUG CLASSIFICATION")
    print("📊 Training Monitor & Results Generator")
    print("=" * 60)
    
    # Check if training is currently running
    if check_training_status():
        print("🔄 Training is currently running...")
        wait_for_training_completion()
    else:
        print("ℹ️  No active training detected. Proceeding to generate results...")
    
    # Generate results
    success = generate_results()
    
    if success:
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Check the results folder for:")
        print("  • Training metrics plots")
        print("  • Comprehensive JSON report")
        print("  • Markdown report")
        print("  • Training results CSV")
        print("=" * 60)
    else:
        print("\n❌ Some tasks failed. Please check the logs.")
    
    return success

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('/home/ubuntu/pharma_system/yolo_model')
    
    # Add current directory to Python path
    sys.path.insert(0, '/home/ubuntu/pharma_system/yolo_model')
    
    success = main()
    sys.exit(0 if success else 1) 