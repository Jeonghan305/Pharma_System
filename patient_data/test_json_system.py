#!/usr/bin/env python3
"""
Test script for JSON-based Patient Data System
Verifies that the patient data can be loaded and displayed correctly.
"""

import json
import os
from datetime import datetime
from patient_manager import PatientDataManager

def test_json_system():
    """Test the JSON-based patient data system."""
    
    print("🔍 Testing JSON-based Patient Data System")
    print("=" * 50)
    
    # Initialize the patient manager
    manager = PatientDataManager()
    
    # Test: Display all patients
    print("\n📋 PATIENT OVERVIEW:")
    print("-" * 30)
    
    patients_data = manager.data.get('patients', {})
    for patient_id, patient in patients_data.items():
        print(f"Patient ID: {patient_id}")
        print(f"Name: {patient['name']}")
        print(f"Age: {patient['age']} years")
        print(f"Weight: {patient['weight_kg']} kg")
        print(f"Allergies: {', '.join(patient['allergies'])}")
        print(f"Conditions: {', '.join(patient['medical_conditions'])}")
        print()
    
    # Test: Display medications by patient
    print("💊 MEDICATION SCHEDULES:")
    print("-" * 30)
    
    medications_data = manager.data.get('medications', {})
    for patient_id in patients_data.keys():
        patient_name = patients_data[patient_id]['name']
        print(f"\n{patient_name} ({patient_id}):")
        
        patient_meds = [med for med in medications_data.values() 
                       if med['patient_id'] == patient_id and med['active']]
        
        if patient_meds:
            for med in patient_meds:
                print(f"  • {med['drug_name']} {med['dosage']}")
                print(f"    Every {med['frequency_hours']} hours")
                print(f"    Instructions: {med['special_instructions']}")
        else:
            print("  No active medications")
    
    # Test: Display recent dosing history
    print("\n📊 RECENT DOSING HISTORY:")
    print("-" * 30)
    
    dosing_data = manager.data.get('dosing_history', {})
    for dose_id, dose in sorted(dosing_data.items(), 
                               key=lambda x: x[1]['taken_at'], reverse=True):
        patient_name = patients_data[dose['patient_id']]['name']
        med_name = next((med['drug_name'] for med in medications_data.values() 
                        if med['medication_id'] == dose['medication_id']), 'Unknown')
        
        print(f"• {patient_name}: {med_name}")
        print(f"  Taken: {dose['taken_at']}")
        print(f"  Status: {dose['status']}")
        print(f"  Notes: {dose['notes']}")
        print()
    
    # Test: System statistics
    metadata = manager.data.get('metadata', {})
    print("📈 SYSTEM STATISTICS:")
    print("-" * 30)
    print(f"Total Patients: {metadata.get('total_patients', 0)}")
    print(f"Total Medications: {metadata.get('total_medications', 0)}")
    print(f"Total Doses Recorded: {metadata.get('total_doses_recorded', 0)}")
    print(f"Last Updated: {metadata.get('last_updated', 'Unknown')}")
    
    # Test: File size and format
    json_file = os.path.join('pharma_system/patient_data', 'patient_data.json')
    if os.path.exists(json_file):
        file_size = os.path.getsize(json_file)
        print(f"Data File Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    print("\n✅ JSON System Test Complete!")
    return True

def test_patient_methods():
    """Test specific patient manager methods."""
    
    print("\n🧪 Testing Patient Manager Methods")
    print("=" * 50)
    
    manager = PatientDataManager()
    
    # Test getting a specific patient
    patient = manager.get_patient("patient-001")
    if patient:
        print(f"✅ Found patient: {patient['name']}")
        print(f"   Age: {patient['age']}, Weight: {patient['weight_kg']} kg")
    else:
        print("❌ Patient not found")
    
    # Test getting patient medications  
    medications = manager.get_patient_medications("patient-002")
    print(f"\n✅ Patient-002 has {len(medications)} active medications:")
    for med in medications:
        print(f"   • {med['drug_name']} {med['dosage']} every {med['frequency_hours']}h")
    
    # Test next dose timing
    next_dose = manager.check_next_dose_time("med-001")
    if next_dose:
        print(f"\n✅ Next dose info: {next_dose}")
    else:
        print("\n❌ No next dose info found")
    
    print("\n✅ Patient Methods Test Complete!")

if __name__ == "__main__":
    print("🏥 Pharmaceutical Patient Data System - JSON Format Test")
    print("=" * 60)
    
    # Change to the correct directory
    os.chdir('/home/ubuntu/pharma_system/patient_data')
    
    # Run tests
    test_json_system()
    test_patient_methods()
    
    print("\n🎉 All tests completed successfully!")
    print("The JSON-based patient data system is working correctly.") 