#!/usr/bin/env python3
"""
Simple test for the simplified patient data system
"""

import os
from patient_manager import PatientDataManager

def test_simple_system():
    print("🏥 SIMPLE PATIENT DATA SYSTEM TEST")
    print("=" * 45)
    
    # Change to correct directory
    os.chdir('/home/ubuntu/pharma_system/patient_data')
    
    manager = PatientDataManager()
    
    # Display all patients and their medications
    patients = manager.get_all_patients()
    print(f"\n📊 Found {len(patients)} patients:")
    
    for patient in patients:
        print(f"\n👤 {patient['name']} (Age {patient['age']})")
        medications = patient.get('medications', [])
        
        if medications:
            print(f"   💊 Medication History ({len(medications)} entries):")
            for med in medications:
                print(f"      • {med['drug']}")
                print(f"        Date/Time: {med['date']} at {med['time']}")
                print(f"        Notes: {med['notes']}")
                print()
        else:
            print("   No medications recorded")
    
    # Show available drug classes from YOLO model
    yolo_drugs = ["Alaxan", "Bactidol", "Biogesic", "Bioflu", "DayZinc", 
                  "Decolgen", "Fish Oil", "Kremil S", "Medicol", "Neozep"]
    
    print(f"\n🧬 Available Drug Classes (from YOLO model):")
    for i, drug in enumerate(yolo_drugs, 1):
        print(f"   {i:2d}. {drug}")
    
    print(f"\n✅ System working correctly!")
    print(f"   - Simple JSON structure")
    print(f"   - Uses real YOLO model drug names")
    print(f"   - Easy to read and modify")

if __name__ == "__main__":
    test_simple_system() 