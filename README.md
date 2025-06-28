# Pharmaceutical Drug Safety Assessment System

A comprehensive AI-powered system that combines computer vision, patient data management, and intelligent assessment to ensure medication safety.

## 🏥 System Overview

This system provides **end-to-end medication safety assessment** through three integrated components:

1. **🤖 YOLO Model** - Drug identification with high accuracy
2. **📋 Patient Data** - Medication schedules and patient information management  
3. **🧠 GPT Assessment** - AI safety evaluation using GPT-4o mini

## 📁 Project Structure

```
pharma_system/
├── yolo_model/              # Drug classification system
│   ├── pre-training/        # Dataset preparation scripts
│   │   ├── prepare_dataset.py
│   │   └── download_dataset.py
│   ├── training/            # Model training scripts
│   │   └── train_pharma_real.py
│   ├── post-training/       # Model evaluation and deployment
│   │   ├── predict.py       # Main prediction script
│   │   ├── generate_report.py
│   │   ├── monitor_and_generate.py
│   │   └── plot_metrics.py
│   ├── data/               # Training and test datasets
│   │   ├── train/          # Training data
│   │   ├── val/            # Validation data
│   │   └── test/           # Test data
│   ├── models/             # Model storage
│   │   ├── trained/        # Trained model weights
│   │   └── pretrained/     # Pre-trained base models
│   ├── outputs/            # Training outputs and reports
│   │   ├── runs/           # Training runs
│   │   ├── plots/          # Performance plots
│   │   ├── reports/        # Evaluation reports
│   │   └── logs/           # Training logs
│   └── README.md           # YOLO model documentation
├── patient_data/           # Patient management system
│   ├── patient_manager.py  # Patient & medication database
│   ├── patient_data.json   # Sample patient data
│   ├── test_simple_system.py
│   └── test_json_system.py
├── outputs/                # System outputs
│   └── assessments/        # Assessment results
├── config_assessment.py    # Main assessment configuration
├── config.yaml            # System configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key for GPT assessment
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 2. Run Assessment

```bash
python config_assessment.py
```

### 3. Use Individual Components

```python
# YOLO Drug Prediction
from yolo_model.post_training.predict import PharmaDrugPredictor
predictor = PharmaDrugPredictor()
result = predictor.predict_drug("path/to/drug/image.jpg")

# Patient Data Management
from patient_data.patient_manager import PatientDataManager
manager = PatientDataManager()
patient_data = manager.load_patient_data()
```

## 💊 Supported Medications

The system can identify **10 pharmaceutical drugs** commonly available in the Philippines:

1. **Alaxan** - Pain reliever and anti-inflammatory
2. **Bactidol** - Antiseptic mouthwash/gargle
3. **Bioflu** - Cold and flu medication
4. **Biogesic** - Pain reliever (Paracetamol)
5. **DayZinc** - Vitamin and mineral supplement
6. **Decolgen** - Cold and cough medication
7. **Fish Oil** - Omega-3 dietary supplement
8. **Kremil S** - Antacid for stomach acidity
9. **Medicol** - Pain reliever and fever reducer
10. **Neozep** - Cold and allergy medication

## 🔄 Complete Workflow

### 1. Drug Identification
- Patient takes photo of medication
- YOLO model identifies drug type with confidence score
- Returns prediction with confidence level

### 2. Patient Data Check
- Retrieves patient information (age, medications, conditions)
- Checks current medication schedule and history
- Analyzes potential drug interactions

### 3. Safety Assessment
- GPT-4o mini analyzes all data using structured prompts
- Considers drug interactions, timing, and patient factors
- Evaluates environment and storage conditions

### 4. Decision Output
- **SAFE**: Ready to take medication
- **WAIT**: Timing not appropriate
- **CAUTION**: Can take with monitoring/precautions
- **CONSULT_DOCTOR**: Seek medical consultation

## 🎯 Key Features

### 🤖 YOLO Model
- **High accuracy** pharmaceutical drug classification
- **Fast inference** for real-time predictions
- **Confidence scoring** for safety verification
- **Support for Filipino pharmaceutical brands**

### 📋 Patient Data System
- **JSON-based** patient information storage
- **Medication history** tracking
- **Flexible data structure** for various patient information
- **Easy integration** with existing systems

### 🧠 GPT Assessment
- **Structured prompts** for consistent analysis
- **Multi-factor safety evaluation**
- **Environmental assessment** capabilities
- **Contextual recommendations**

## ⚙️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Configuration File (config.yaml)
```yaml
# System configuration
yolo_model:
  threshold: 0.7
  model_path: "models/trained/best_pharma_yolov8.pt"

gpt_assessment:
  model: "gpt-4o-mini"
  temperature: 0.1

patient_data:
  source: "patient_data/patient_data.json"
```

## 🔒 Safety & Privacy

- **Local Processing**: YOLO runs locally, no image upload required
- **Secure Storage**: Patient data handled with privacy considerations
- **Audit Trail**: All assessments logged with timestamps
- **Medical Disclaimer**: Educational/research purposes only

## 📈 Performance Metrics

- **Model Training**: Optimized for Philippine pharmaceutical drugs
- **Inference Speed**: Fast local predictions
- **Accuracy**: High confidence scoring system
- **Memory Efficient**: Optimized for deployment

## 🚨 Important Notes

⚠️ **Medical Disclaimer**: This system is for educational/research purposes. Always consult healthcare professionals for medical decisions.

⚠️ **API Requirements**: GPT assessment requires OpenAI API access.

⚠️ **Image Quality**: Ensure clear, well-lit photos for best results.

⚠️ **Local Context**: Optimized for Philippine pharmaceutical market.

## 🔧 Development

### Training New Models
1. Prepare dataset using `pre-training/prepare_dataset.py`
2. Train model with `training/train_pharma_real.py`
3. Evaluate using `post-training/generate_report.py`

### Adding New Drugs
1. Update training dataset with new drug images
2. Retrain YOLO model with updated classes
3. Update drug mapping in prediction scripts

### Customizing Assessments
1. Modify GPT prompts in `config_assessment.py`
2. Adjust safety thresholds and risk levels
3. Add custom patient condition handling

## 📞 Support

For issues or questions:
1. Check the configuration file setup
2. Verify API key configuration
3. Review model file paths
4. Check patient data format

---

**Built with**: YOLOv8, PyTorch, OpenAI GPT-4o mini, Python  
**Focused on**: Philippine pharmaceutical market  
**Last Updated**: June 2025
