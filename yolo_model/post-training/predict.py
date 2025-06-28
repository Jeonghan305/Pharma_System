#!/usr/bin/env python3
"""
Pharmaceutical Drug Classification Predictor
Predicts drug type from image using trained YOLOv8 model
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json

class PharmaDrugPredictor:
    def __init__(self, model_path="weights/best.pt"):
        """Initialize the drug classification predictor"""
        self.model_path = Path(__file__).parent / model_path
        self.model = YOLO(self.model_path)
        
        # Drug classes mapping
        self.drug_classes = {
            0: "Adderall",
            1: "Benztropine", 
            2: "Betamethasone",
            3: "Ezetimibe",
            4: "Glimepiride",
            5: "Lisinopril",
            6: "Loperamide",
            7: "Rivaroxaban",
            8: "Tramadol",
            9: "Warfarin"
        }
        
    def predict_drug(self, image_path, confidence_threshold=0.7):
        """
        Predict drug type from image
        
        Args:
            image_path: Path to the drug image
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            dict: Prediction results with drug name, confidence, and metadata
        """
        try:
            # Run prediction
            results = self.model(image_path, verbose=False)
            result = results[0]
            
            # Get top prediction
            probs = result.probs
            top_class = probs.top1
            top_confidence = probs.top1conf.item()
            top5_classes = probs.top5
            top5_confidences = probs.top5conf.tolist()
            
            # Build prediction result
            prediction = {
                "image_path": str(image_path),
                "predicted_drug": self.drug_classes[top_class],
                "confidence": round(top_confidence, 4),
                "meets_threshold": top_confidence >= confidence_threshold,
                "top5_predictions": [
                    {
                        "drug": self.drug_classes[cls],
                        "confidence": round(conf, 4)
                    }
                    for cls, conf in zip(top5_classes, top5_confidences)
                ],
                "model_info": {
                    "model_path": str(self.model_path),
                    "confidence_threshold": confidence_threshold
                }
            }
            
            return prediction
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "image_path": str(image_path)
            }
    
    def batch_predict(self, image_paths, confidence_threshold=0.7):
        """Predict multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict_drug(image_path, confidence_threshold)
            results.append(result)
        return results

def main():
    """Example usage"""
    predictor = PharmaDrugPredictor()
    
    # Example prediction
    # result = predictor.predict_drug("path/to/drug/image.jpg")
    # print(json.dumps(result, indent=2))
    
    print("Pharmaceutical Drug Predictor initialized successfully!")
    print(f"Model loaded from: {predictor.model_path}")
    print(f"Available drug classes: {list(predictor.drug_classes.values())}")

if __name__ == "__main__":
    main() 