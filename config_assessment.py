#!/usr/bin/env python3
"""
Configuration-Driven Pharmaceutical Drug Safety Assessment
Reads settings from config.yaml for flexible testing scenarios
"""

import yaml
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add yolo_model to path for imports
sys.path.append(str(Path(__file__).parent / "yolo_model"))

from ultralytics import YOLO
# Remove broken import: from medication_assessor import MedicationAssessor

# Add OpenAI import for direct GPT calls
import openai


class ConfigurableAssessment:
    """Config-driven medication safety assessment"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration file"""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_paths()
        
    def load_config(self):
        """Load YAML configuration"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"✅ Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
            sys.exit(1)
    
    def setup_paths(self):
        """Setup and validate file paths"""
        base_dir = Path(__file__).parent
        
        # Patient data path - use absolute path from config
        patient_data_path_str = self.config['patient_data']['database_path']
        self.patient_data_path = Path(patient_data_path_str) if patient_data_path_str.startswith('/') else base_dir / patient_data_path_str
        
        # YOLO model path - use absolute path from config  
        model_path_str = self.config['yolo_model']['model_path']
        self.model_path = Path(model_path_str) if model_path_str.startswith('/') else base_dir / model_path_str
        
        # Output directory
        self.output_dir = base_dir / self.config['output']['reports_directory']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate paths exist
        if not self.patient_data_path.exists():
            print(f"❌ Patient data not found: {self.patient_data_path}")
            sys.exit(1)
            
        if not self.model_path.exists():
            print(f"❌ YOLO model not found: {self.model_path}")
            sys.exit(1)
    
    def load_patient_data(self):
        """Load patient database"""
        try:
            with open(self.patient_data_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"❌ Error loading patient data: {e}")
            sys.exit(1)
    
    def get_patient_info(self, patient_id):
        """Get specific patient information"""
        patients = self.load_patient_data()
        
        for patient in patients['patients']:
            if patient['id'] == patient_id:
                return patient
                
        print(f"❌ Patient not found: {patient_id}")
        print(f"Available patients: {[p['id'] for p in patients['patients']]}")
        sys.exit(1)
    
    def run_yolo_prediction(self, image_path):
        """Run YOLO drug identification"""
        try:
            print(f"🔍 Loading YOLO model from {self.model_path}")
            model = YOLO(str(self.model_path))
            
            print(f"📸 Analyzing image: {image_path}")
            results = model(image_path, conf=self.config['yolo_model']['confidence_threshold'])
            
            # Extract prediction details
            if results and len(results) > 0:
                result = results[0]
                if result.probs is not None:
                    # Get top prediction
                    top_class_idx = result.probs.top1
                    confidence = result.probs.top1conf.item()
                    class_name = result.names[top_class_idx]
                    
                    prediction = {
                        'predicted_drug': class_name,
                        'confidence': confidence,
                        'model_path': str(self.model_path),
                        'threshold': self.config['yolo_model']['confidence_threshold']
                    }
                    
                    print(f"🔬 YOLO Prediction: {class_name} ({confidence:.1%} confidence)")
                    return prediction
            
            print("❌ No drug detected in image")
            return None
            
        except Exception as e:
            print(f"❌ YOLO prediction error: {e}")
            return None
    
    def run_gpt_assessment(self, patient_info, yolo_prediction, image_path):
        """Run simplified GPT assessment focused only on medication timing"""
        try:
            print("🤖 Running simplified medication timing assessment...")
            
            # Set up OpenAI API key (priority: env var > config file)
            api_key = os.getenv('OPENAI_API_KEY')  # Environment variable takes priority
            if not api_key:
                api_key = self.config['gpt_assessment'].get('api_key')  # Fallback to config file
                
            if not api_key or api_key == "your-openai-api-key-here":
                print("⚠️ No valid OpenAI API key found - skipping GPT assessment")
                print("💡 Set API key in config.yaml or OPENAI_API_KEY environment variable")
                return {
                    "status": "skipped",
                    "reason": "No valid OpenAI API key provided",
                    "final_recommendation": {
                        "action": "CONSULT_DOCTOR",
                        "reasoning": "No API key - cannot assess timing"
                    }
                }
            
            # Import OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Prepare patient context
            drug_identified = yolo_prediction['predicted_drug']
            patient_medications = patient_info.get('medications', [])
            patient_name = patient_info.get('name')
            
            # Create simple timing assessment prompt
            system_prompt = """You are a medication timing assistant. Your only job is to determine if a patient should take their medication based on timing.

RULES:
- Patients should take medications every 8 hours
- If patient doesn't take this medication, say they don't take it
- If patient takes it and last dose was >8 hours ago, recommend taking it
- If patient takes it and last dose was <8 hours ago, recommend waiting

RESPOND IN THIS EXACT JSON FORMAT:
{
    "patient_takes_medication": true/false,
    "last_taken": "YYYY-MM-DD HH:MM" or null,
    "hours_since_last_dose": number or null,
    "action": "TAKE_NOW/WAIT/PATIENT_DOESNT_TAKE",
    "reasoning": "Brief explanation",
    "wait_time_remaining": "X hours remaining" or null
}"""
            
            user_prompt = f"""
MEDICATION TIMING CHECK:

Patient: {patient_name}
Identified Medication: {drug_identified}

Patient's Medication History:
{json.dumps(patient_medications, indent=2)}

Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Task: Check if patient takes {drug_identified}, and if so, determine timing recommendation based on 8-hour intervals.
            """
            
            # Call GPT-4o-mini (no image needed for timing check)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            gpt_response = response.choices[0].message.content.strip()
            print(f"📋 Timing analysis completed")
            
            # Try to parse JSON response
            try:
                assessment = json.loads(gpt_response)
                
                # Add metadata
                assessment["metadata"] = {
                    "model_used": "gpt-4o-mini",
                    "analysis_timestamp": datetime.now().isoformat(),
                    "assessment_type": "timing_only"
                }
                
                # Print summary
                action = assessment.get("action", "CONSULT_DOCTOR")
                reasoning = assessment.get("reasoning", "No reasoning provided")
                print(f"🎯 Decision: {action} - {reasoning}")
                
                return assessment
                
            except json.JSONDecodeError:
                print("⚠️ GPT response was not valid JSON, using fallback")
                return {
                    "status": "parsing_error",
                    "raw_response": gpt_response,
                    "final_recommendation": {
                        "action": "CONSULT_DOCTOR",
                        "reasoning": "AI timing analysis format error"
                    }
                }
                
        except Exception as e:
            print(f"❌ GPT assessment error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "final_recommendation": {
                    "action": "CONSULT_DOCTOR",
                    "reasoning": f"Assessment failed: {str(e)}"
                }
            }
    
    def generate_assessment_report(self, patient_info, yolo_prediction, gpt_assessment):
        """Generate comprehensive assessment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'metadata': {
                'timestamp': timestamp,
                'scenario_name': self.config['assessment'].get('scenario_name', 'Pharmaceutical Safety Assessment'),
                'notes': self.config['assessment'].get('notes', 'Configuration-driven assessment'),
                'config_file': self.config_path
            },
            'input': {
                'image_path': self.config['assessment']['image_path'],
                'patient_id': self.config['assessment']['patient_id'],
                'expected_drug': self.config['assessment'].get('expected_drug', 'Unknown')
            },
            'patient_data': patient_info,
            'yolo_results': yolo_prediction,
            'gpt_assessment': gpt_assessment,
            'verification': {
                'prediction_matches_expected': (
                    yolo_prediction['predicted_drug'] == self.config['assessment'].get('expected_drug')
                    if yolo_prediction and self.config['assessment'].get('expected_drug') else False
                ),
                'patient_takes_this_drug': (
                    yolo_prediction['predicted_drug'] in patient_info.get('medication_history', [])
                    if yolo_prediction else False
                )
            }
        }
        
        return report
    
    def save_results(self, report):
        """Save assessment results"""
        timestamp = report['metadata']['timestamp']
        patient_id = report['input']['patient_id']
        
        # Save JSON report
        if self.config['output'].get('save_json', True):  # Default to True
            json_path = self.output_dir / f"assessment_{patient_id}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Results saved to: {json_path}")
        
        # Save summary
        if self.config['output'].get('save_summary', True):  # Default to True
            summary_path = self.output_dir / f"summary_{patient_id}_{timestamp}.txt"
            with open(summary_path, 'w') as f:
                self.write_summary(f, report)
            print(f"📋 Summary saved to: {summary_path}")
    
    def write_summary(self, file, report):
        """Write human-readable summary"""
        file.write("=" * 60 + "\n")
        file.write("PHARMACEUTICAL MEDICATION TIMING ASSESSMENT REPORT\n")
        file.write("=" * 60 + "\n\n")
        
        # Metadata
        file.write(f"Assessment: {report['metadata']['scenario_name']}\n")
        file.write(f"Timestamp: {report['metadata']['timestamp']}\n")
        file.write(f"Notes: {report['metadata']['notes']}\n\n")
        
        # Input
        file.write("INPUT CONFIGURATION:\n")
        file.write(f"  Image Path: {report['input']['image_path']}\n")
        file.write(f"  Patient ID: {report['input']['patient_id']}\n\n")
        
        # Patient info
        patient = report['patient_data']
        file.write("PATIENT INFORMATION:\n")
        file.write(f"  Name: {patient.get('name', 'N/A')}\n")
        file.write(f"  Age: {patient.get('age', 'N/A')}\n\n")
        
        # YOLO results
        if report['yolo_results']:
            yolo = report['yolo_results']
            file.write("YOLO DRUG IDENTIFICATION:\n")
            file.write(f"  Predicted Drug: {yolo['predicted_drug']}\n")
            file.write(f"  Confidence: {yolo['confidence']:.1%}\n\n")
        
        # GPT timing assessment
        if report['gpt_assessment']:
            gpt = report['gpt_assessment']
            file.write("MEDICATION TIMING ASSESSMENT:\n")
            
            # Check if simplified format
            if 'patient_takes_medication' in gpt:
                takes_med = gpt.get('patient_takes_medication', False)
                action = gpt.get('action', 'CONSULT_DOCTOR')
                reasoning = gpt.get('reasoning', 'N/A')
                
                file.write(f"  Patient takes this medication: {'YES' if takes_med else 'NO'}\n")
                
                if takes_med:
                    last_taken = gpt.get('last_taken')
                    hours_since = gpt.get('hours_since_last_dose')
                    wait_time = gpt.get('wait_time_remaining')
                    
                    if last_taken:
                        file.write(f"  Last taken: {last_taken}\n")
                    if hours_since is not None:
                        file.write(f"  Hours since last dose: {hours_since}\n")
                    if wait_time:
                        file.write(f"  Wait time remaining: {wait_time}\n")
                
                file.write(f"  RECOMMENDATION: {action}\n")
                file.write(f"  Reasoning: {reasoning}\n\n")
            
            # Fallback for error cases
            elif gpt.get('status') in ['error', 'skipped', 'parsing_error']:
                file.write(f"  Status: {gpt.get('status', 'Unknown')}\n")
                if 'final_recommendation' in gpt:
                    rec = gpt['final_recommendation']
                    file.write(f"  Action: {rec.get('action', 'CONSULT_DOCTOR')}\n")
                    file.write(f"  Reasoning: {rec.get('reasoning', 'N/A')}\n")
    
    def print_detailed_results(self, report):
        """Print detailed results to console"""
        if not self.config['output'].get('print_detailed', True):  # Default to True
            return
            
        print("\n" + "=" * 60)
        print("🏥 PHARMACEUTICAL MEDICATION TIMING ASSESSMENT")
        print("=" * 60)
        
        print(f"\n📋 Scenario: {report['metadata']['scenario_name']}")
        print(f"🕒 Timestamp: {report['metadata']['timestamp']}")
        
        # Patient info
        patient = report['patient_data']
        print(f"\n👤 Patient: {patient.get('name')} (ID: {patient.get('id')})")
        print(f"🎂 Age: {patient.get('age')}")
        
        # YOLO results
        if report['yolo_results']:
            yolo = report['yolo_results']
            print(f"\n🔬 YOLO Prediction: {yolo['predicted_drug']}")
            print(f"📊 Confidence: {yolo['confidence']:.1%}")
        
        # GPT timing assessment
        if report['gpt_assessment']:
            gpt = report['gpt_assessment']
            print(f"\n🤖 Medication Timing Assessment:")
            
            # Check if simplified format
            if 'patient_takes_medication' in gpt:
                takes_med = gpt.get('patient_takes_medication', False)
                action = gpt.get('action', 'CONSULT_DOCTOR')
                reasoning = gpt.get('reasoning', 'N/A')
                
                print(f"   💊 Patient takes this medication: {'YES' if takes_med else 'NO'}")
                
                if takes_med:
                    last_taken = gpt.get('last_taken')
                    hours_since = gpt.get('hours_since_last_dose')
                    wait_time = gpt.get('wait_time_remaining')
                    
                    if last_taken:
                        print(f"   🕒 Last taken: {last_taken}")
                    if hours_since is not None:
                        print(f"   ⏰ Hours since last dose: {hours_since}")
                    if wait_time:
                        print(f"   ⏳ Wait time remaining: {wait_time}")
                
                # Color code the action
                if action == 'TAKE_NOW':
                    print(f"   🟢 FINAL DECISION: {action}")
                elif action == 'WAIT':
                    print(f"   🟡 FINAL DECISION: {action}")
                elif action == 'PATIENT_DOESNT_TAKE':
                    print(f"   ⚪ FINAL DECISION: {action}")
                else:
                    print(f"   🔵 FINAL DECISION: {action}")
                
                print(f"   💡 Reasoning: {reasoning}")
            
            # Fallback for error cases
            elif gpt.get('status') in ['error', 'skipped', 'parsing_error']:
                print(f"   ⚠️ Status: {gpt.get('status', 'Unknown')}")
                if 'final_recommendation' in gpt:
                    rec = gpt['final_recommendation']
                    print(f"   🔵 Action: {rec.get('action', 'CONSULT_DOCTOR')}")
                    print(f"   💡 Reason: {rec.get('reasoning', 'N/A')}")
        
        print("\n" + "=" * 60)
    
    def run_assessment(self):
        """Run complete assessment workflow"""
        print("🚀 Starting Configuration-Driven Assessment")
        print(f"📁 Config: {self.config_path}")
        
        # Get assessment parameters
        image_path = self.config['assessment']['image_path']
        patient_id = self.config['assessment']['patient_id']
        
        # Validate image path
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            print("Please update the 'assessment.image_path' in config.yaml")
            return None
        
        # Load patient data
        print(f"👤 Loading patient data for: {patient_id}")
        patient_info = self.get_patient_info(patient_id)
        
        # Run YOLO prediction
        yolo_prediction = self.run_yolo_prediction(image_path)
        if not yolo_prediction:
            print("❌ YOLO prediction failed - cannot continue assessment")
            return None
        
        # Run GPT assessment
        gpt_assessment = self.run_gpt_assessment(patient_info, yolo_prediction, image_path)
        
        # Generate report
        report = self.generate_assessment_report(patient_info, yolo_prediction, gpt_assessment)
        
        # Output results
        self.print_detailed_results(report)
        self.save_results(report)
        
        print("✅ Assessment completed successfully!")
        return report


def main():
    """Main execution function"""
    print("🏥 Configuration-Driven Pharmaceutical Assessment System")
    print("=" * 50)
    
    # Initialize assessment
    assessment = ConfigurableAssessment()
    
    # Run assessment
    result = assessment.run_assessment()
    
    if result:
        print(f"\n📊 Assessment completed for patient: {result['input']['patient_id']}")
        print(f"🔬 Drug identified: {result['yolo_results']['predicted_drug']}")
        print(f"📁 Results saved to: {assessment.output_dir}")
    else:
        print("❌ Assessment failed")


if __name__ == "__main__":
    main() 