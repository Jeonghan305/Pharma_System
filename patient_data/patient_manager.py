#!/usr/bin/env python3
"""
Simple Patient Data Management System
Manages basic patient medication history using simplified JSON storage
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class PatientDataManager:
    def __init__(self, data_file="patient_data.json"):
        """Initialize patient data manager with JSON storage"""
        self.data_file = Path(__file__).parent / data_file
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load data from JSON file"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        else:
            return {"patients": []}
    
    def save_data(self):
        """Save data to JSON file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        return self.data.get("patients", [])
    
    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient by ID"""
        for patient in self.data.get("patients", []):
            if patient["id"] == patient_id:
                return patient
        return None
    
    def get_patient_medications(self, patient_id: str) -> List[Dict]:
        """Get all medications for a patient"""
        patient = self.get_patient(patient_id)
        if patient:
            return patient.get("medications", [])
        return []
    
    def add_medication(self, patient_id: str, drug: str, date: str, time: str, notes: str = ""):
        """Add a medication entry for a patient"""
        patient = self.get_patient(patient_id)
        if patient:
            if "medications" not in patient:
                patient["medications"] = []
            
            patient["medications"].append({
                "drug": drug,
                "date": date,
                "time": time,
                "notes": notes
            })
            self.save_data()
            return True
        return False

def main():
    """Simple demo"""
    manager = PatientDataManager()
    
    print("📋 Simple Patient Data System")
    print("=" * 40)
    
    patients = manager.get_all_patients()
    print(f"Total patients: {len(patients)}")
    
    for patient in patients:
        print(f"\n👤 {patient['name']} (Age: {patient['age']})")
        medications = patient.get('medications', [])
        print(f"   Medications taken: {len(medications)}")
        
        for med in medications:
            print(f"   • {med['drug']} on {med['date']} at {med['time']}")
            if med['notes']:
                print(f"     Notes: {med['notes']}")

if __name__ == "__main__":
    main() 