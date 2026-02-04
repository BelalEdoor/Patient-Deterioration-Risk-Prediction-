"""
تشخيص مشكلة النموذج
Model Diagnosis Script
"""

# مريض بقيم خطيرة جداً (يجب أن يعطي High Risk)
critical_patient = {
    'patient_id': 'a245',
    'age': 87,
    'gender': 1,  # Male
    'heart_rate': 120,  # عالي جداً
    'respiratory_rate': 25,  # عالي
    'spo2_pct': 80,  # منخفض جداً 🚨
    'temperature_c': 40,  # حمى شديدة 🚨
    'systolic_bp': 150,
    'diastolic_bp': 110,  # عالي جداً
    'wbc_count': 15,  # عالي
    'lactate': 4.0,  # عالي جداً 🚨
    'creatinine': 2.0,  # مرتفع
    'crp_level': 15,  # مرتفع
    'hemoglobin': 20,
    'oxygen_flow': 8,  # يحتاج أكسجين
    'oxygen_device': 1,  # Nasal Cannula
    'nurse_alert': 1,  # Yes
    'mobility_score': 1,  # Chair/Wheelchair
    'comorbidity_index': 2
}

# مريض بقيم طبيعية تماماً (يجب أن يعطي Low Risk)
healthy_patient = {
    'patient_id': 'p100',
    'age': 45,
    'gender': 1,
    'heart_rate': 75,
    'respiratory_rate': 16,
    'spo2_pct': 98,
    'temperature_c': 36.8,
    'systolic_bp': 115,
    'diastolic_bp': 75,
    'wbc_count': 7.5,
    'lactate': 1.2,
    'creatinine': 1.0,
    'crp_level': 3,
    'hemoglobin': 14.5,
    'oxygen_flow': 0,
    'oxygen_device': 0,
    'nurse_alert': 0,
    'mobility_score': 4,  # Fully mobile
    'comorbidity_index': 0
}

print("="*70)
print("🔍 MODEL DIAGNOSIS TEST")
print("="*70)

from ML_Model import PatientRiskModel

model = PatientRiskModel()

if model.is_trained():
    print("\n✅ Model loaded successfully")
    
    print("\n" + "="*70)
    print("TEST 1: CRITICAL PATIENT (Should be HIGH RISK)")
    print("="*70)
    print("Input values:")
    print(f"  Age: 87 (elderly)")
    print(f"  SpO2: 80% 🚨 (CRITICAL - Normal: 95-100%)")
    print(f"  Temperature: 40°C 🚨 (SEVERE FEVER - Normal: 36-37°C)")
    print(f"  Heart Rate: 120 bpm 🚨 (HIGH - Normal: 60-100)")
    print(f"  Lactate: 4.0 🚨 (VERY HIGH - Normal: 0.5-2.0)")
    print(f"  Respiratory Rate: 25 🚨 (HIGH - Normal: 12-20)")
    print(f"  Diastolic BP: 110 🚨 (HIGH - Normal: 60-80)")
    print(f"  WBC: 15 🚨 (HIGH - Normal: 4-11)")
    print(f"  Oxygen Flow: 8 L/min (needs oxygen support)")
    print(f"  Nurse Alert: YES")
    
    result1 = model.predict(critical_patient)
    
    print(f"\n📊 Model Prediction:")
    print(f"  Risk Score: {result1['risk_score']*100:.1f}%")
    print(f"  Risk Category: {result1['risk_category']}")
    print(f"  Prediction: {'Deterioration' if result1['prediction'] == 1 else 'Stable'}")
    
    if result1['risk_score'] < 0.5:
        print("\n❌ ERROR: Model predicts LOW RISK for critically ill patient!")
        print("   This is a SERIOUS MODEL FAILURE!")
    else:
        print("\n✅ Model correctly identifies high risk")
    
    print("\n" + "="*70)
    print("TEST 2: HEALTHY PATIENT (Should be LOW RISK)")
    print("="*70)
    print("Input values:")
    print(f"  Age: 45 (middle-aged)")
    print(f"  All vital signs: NORMAL ✅")
    print(f"  No oxygen support")
    print(f"  Fully mobile")
    print(f"  No alerts")
    
    result2 = model.predict(healthy_patient)
    
    print(f"\n📊 Model Prediction:")
    print(f"  Risk Score: {result2['risk_score']*100:.1f}%")
    print(f"  Risk Category: {result2['risk_category']}")
    
    if result2['risk_score'] > 0.5:
        print("\n⚠️ WARNING: Model predicts HIGH RISK for healthy patient!")
    else:
        print("\n✅ Model correctly identifies low risk")
    
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    if result1['risk_score'] < 0.5:
        print("\n🚨 CRITICAL ISSUE DETECTED:")
        print("   The model is NOT working correctly!")
        print("\n   Possible causes:")
        print("   1. Model was trained on incorrect/imbalanced data")
        print("   2. Feature scaling is inverted")
        print("   3. Model architecture issue")
        print("   4. Training data has wrong labels")
        print("\n   ⚠️ DO NOT USE THIS MODEL IN PRODUCTION!")
        print("   It could miss critically ill patients!")
        
        print("\n🔧 RECOMMENDED FIXES:")
        print("   1. Re-check training data labels")
        print("   2. Verify feature engineering")
        print("   3. Check if target variable is inverted")
        print("   4. Re-train model with verified data")
    else:
        print("\n✅ Model appears to be functioning correctly")
    
else:
    print("❌ Model not trained!")

print("\n" + "="*70)