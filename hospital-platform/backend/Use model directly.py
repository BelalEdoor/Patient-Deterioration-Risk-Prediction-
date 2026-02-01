"""
استخدام نموذج الشبكة العصبية المدرب بشكل مباشر (بدون API)
Direct Neural Network Model Usage (without API)
"""

import numpy as np
import pandas as pd
from ML_Model import PatientRiskModel, get_clinical_recommendations

# ===== 1. تحميل النموذج المدرب =====
# Load the trained Neural Network model

print("\n" + "="*60)
print("🧠 Neural Network Model - Direct Usage")
print("="*60 + "\n")

model = PatientRiskModel(model_dir="models")

if not model.is_trained():
    print("❌ Error: Model not trained yet!")
    print("\nPlease train the model first:")
    print("  1. Start the backend: python main.py")
    print("  2. Go to http://localhost:3000")
    print("  3. Navigate to 'Train Model' page")
    print("  4. Upload sample_data.csv")
    print("  5. Click 'Start Training'")
    print("\nOr train programmatically:")
    print("  model.train(data_path='../sample_data.csv')")
    exit()

print("✅ Neural Network model loaded successfully!")

# Get model info
info = model.get_model_info()
print(f"\n📊 Model Information:")
print(f"   Type: {info['model_type']}")
print(f"   Architecture: {info['architecture']}")
print(f"   Total Layers: {info['layers']}")
print(f"   Total Parameters: {info['total_params']:,}")
print(f"   Input Features: {info['n_features']}")


# ===== 2. تحضير بيانات مريض =====
# Prepare patient data

def predict_patient(patient_data_dict):
    """
    التنبؤ بخطر تدهور مريض
    Predict patient deterioration risk
    
    Args:
        patient_data_dict: قاموس يحتوي على بيانات المريض
        
    Returns:
        result: نتيجة التنبؤ مع التوصيات
    """
    # Get prediction from Neural Network
    result = model.predict(patient_data_dict)
    
    # Get clinical recommendations
    recommendations = get_clinical_recommendations(
        patient_data_dict, 
        result['risk_score']
    )
    
    result['recommendations'] = recommendations
    return result


# ===== 3. أمثلة على الاستخدام =====
# Usage examples

if __name__ == "__main__":
    
    # بيانات مريض تجريبي - حالة مستقرة
    # Sample patient - stable condition
    stable_patient = {
        'age': 58,
        'gender': 1,  # Male
        'heart_rate': 88,
        'respiratory_rate': 16,
        'spo2_pct': 98,
        'temperature_c': 36.8,
        'systolic_bp': 125,
        'diastolic_bp': 75,
        'wbc_count': 7.2,
        'lactate': 0.9,
        'creatinine': 0.95,
        'crp_level': 3,
        'hemoglobin': 14.2,
        'oxygen_flow': 0,
        'oxygen_device': 0,  # No oxygen
        'nurse_alert': 0,
        'mobility_score': 4,  # Fully mobile
        'comorbidity_index': 1
    }
    
    # بيانات مريض تجريبي - حالة خطرة
    # Sample patient - high risk condition
    high_risk_patient = {
        'age': 81,
        'gender': 0,  # Female
        'heart_rate': 125,
        'respiratory_rate': 28,
        'spo2_pct': 88,
        'temperature_c': 38.8,
        'systolic_bp': 160,
        'diastolic_bp': 95,
        'wbc_count': 15.5,
        'lactate': 3.5,
        'creatinine': 2.2,
        'crp_level': 78,
        'hemoglobin': 9.8,
        'oxygen_flow': 6,
        'oxygen_device': 3,  # Ventilator
        'nurse_alert': 1,
        'mobility_score': 0,  # Bedridden
        'comorbidity_index': 5
    }
    
    print("\n" + "="*60)
    print("🟢 PATIENT 1: Stable Condition / مريض مستقر")
    print("="*60)
    print(f"Age: {stable_patient['age']}, Gender: {'Male' if stable_patient['gender'] else 'Female'}")
    print(f"HR: {stable_patient['heart_rate']}, RR: {stable_patient['respiratory_rate']}, SpO2: {stable_patient['spo2_pct']}%")
    print(f"BP: {stable_patient['systolic_bp']}/{stable_patient['diastolic_bp']}, Temp: {stable_patient['temperature_c']}°C")
    print("-" * 60)
    
    result1 = predict_patient(stable_patient)
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Risk Score: {result1['risk_score']*100:.1f}%")
    print(f"   Category: {result1['risk_category']}")
    print(f"   Prediction: {'⚠️  Deterioration Expected' if result1['prediction'] == 1 else '✅ Stable'}")
    print(f"   Confidence: {result1['confidence']*100:.1f}%")
    print(f"\n📋 Clinical Recommendations:")
    for rec in result1['recommendations']:
        print(f"   • {rec}")
    
    print("\n" + "="*60)
    print("🔴 PATIENT 2: High Risk Condition / مريض في خطر")
    print("="*60)
    print(f"Age: {high_risk_patient['age']}, Gender: {'Male' if high_risk_patient['gender'] else 'Female'}")
    print(f"HR: {high_risk_patient['heart_rate']}, RR: {high_risk_patient['respiratory_rate']}, SpO2: {high_risk_patient['spo2_pct']}%")
    print(f"BP: {high_risk_patient['systolic_bp']}/{high_risk_patient['diastolic_bp']}, Temp: {high_risk_patient['temperature_c']}°C")
    print(f"Lactate: {high_risk_patient['lactate']}, Creatinine: {high_risk_patient['creatinine']}")
    print("-" * 60)
    
    result2 = predict_patient(high_risk_patient)
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Risk Score: {result2['risk_score']*100:.1f}%")
    print(f"   Category: {result2['risk_category']}")
    print(f"   Prediction: {'⚠️  Deterioration Expected' if result2['prediction'] == 1 else '✅ Stable'}")
    print(f"   Confidence: {result2['confidence']*100:.1f}%")
    print(f"\n📋 Clinical Recommendations:")
    for rec in result2['recommendations']:
        print(f"   • {rec}")
    
    
    # ===== 4. التنبؤ لعدة مرضى (Batch) =====
    print("\n" + "="*60)
    print("📊 BATCH PREDICTION - Multiple Patients / عدة مرضى")
    print("="*60)
    
    # قائمة مرضى
    patients = [stable_patient, high_risk_patient]
    patient_names = ["Stable Patient", "High Risk Patient"]
    
    for i, (patient, name) in enumerate(zip(patients, patient_names), 1):
        result = model.predict(patient)
        print(f"\nPatient {i} ({name}):")
        print(f"  Risk Score: {result['risk_score']*100:.1f}%")
        print(f"  Category: {result['risk_category']}")
        print(f"  Prediction: {'Deterioration' if result['prediction'] == 1 else 'Stable'}")
    
    
    # ===== 5. استخدام DataFrame =====
    print("\n" + "="*60)
    print("📋 USING PANDAS DATAFRAME")
    print("="*60)
    
    # إنشاء DataFrame من عدة مرضى
    df_patients = pd.DataFrame([stable_patient, high_risk_patient])
    
    print(f"\nDataFrame shape: {df_patients.shape}")
    print(f"Columns: {len(df_patients.columns)}")
    
    # Batch prediction
    results = model.predict_batch(df_patients)
    
    # إضافة النتائج للـ DataFrame
    df_patients['risk_score'] = [r['risk_score'] for r in results]
    df_patients['prediction'] = [r['prediction'] for r in results]
    df_patients['risk_category'] = [r['risk_category'] for r in results]
    
    print(f"\nPredictions Summary:")
    print(df_patients[['age', 'heart_rate', 'spo2_pct', 'risk_score', 'risk_category', 'prediction']])
    
    
    # ===== 6. إحصائيات التنبؤ =====
    print("\n" + "="*60)
    print("📈 PREDICTION STATISTICS")
    print("="*60)
    
    risk_scores = [r['risk_score'] for r in results]
    predictions = [r['prediction'] for r in results]
    
    print(f"\nTotal Patients Analyzed: {len(results)}")
    print(f"Average Risk Score: {np.mean(risk_scores)*100:.1f}%")
    print(f"Patients predicted to deteriorate: {sum(predictions)}")
    print(f"Patients predicted stable: {len(predictions) - sum(predictions)}")
    
    # Risk distribution
    high_risk = sum(1 for r in results if r['risk_category'] == 'High Risk')
    medium_risk = sum(1 for r in results if r['risk_category'] == 'Medium Risk')
    low_risk = sum(1 for r in results if r['risk_category'] == 'Low Risk')
    
    print(f"\nRisk Category Distribution:")
    print(f"  🔴 High Risk: {high_risk}")
    print(f"  🟡 Medium Risk: {medium_risk}")
    print(f"  🟢 Low Risk: {low_risk}")
    
    
    # ===== 7. حفظ النتائج =====
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    # حفظ إلى CSV
    output_file = "predictions_output.csv"
    df_patients.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    
    print("\n" + "="*60)
    print("🎉 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n💡 Tips:")
    print("  • Use model.predict(patient_dict) for single predictions")
    print("  • Use model.predict_batch(df) for multiple patients")
    print("  • Risk scores range from 0.0 (stable) to 1.0 (high risk)")
    print("  • Recall/Sensitivity is prioritized to minimize false negatives")
    print("  • Neural Network provides probabilistic predictions")
    print("\n🧠 Model Type: Neural Network (Deep Learning)")
    print("   - 4 hidden layers with dropout and batch normalization")
    print("   - Optimized for high recall (catching deteriorating patients)")
    print("   - Trained with class weights to handle imbalanced data")
    print("\n" + "="*60 + "\n")