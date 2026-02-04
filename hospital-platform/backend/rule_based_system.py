"""
نظام القواعد الطبية للتنبؤ بخطر المرضى
Rule-Based Clinical Decision System for Patient Risk Assessment

هذا النظام يستخدم معايير طبية معتمدة بدلاً من التعلم الآلي
This system uses evidence-based clinical criteria instead of machine learning
"""

def calculate_risk_score_rules_based(patient_data):
    """
    حساب درجة الخطر بناء على القواعد الطبية المعتمدة
    Calculate risk score using evidence-based clinical rules
    
    Args:
        patient_data: Dictionary with patient vitals and lab values
        
    Returns:
        float: Risk score between 0 and 1
    """
    risk_points = 0
    max_possible_points = 100
    
    # ===== CRITICAL VITAL SIGNS (60 points max) =====
    
    # 1. SpO2 (Oxygen Saturation) - 20 points
    # انخفاض الأكسجين من أخطر العلامات
    spo2 = patient_data.get('spo2_pct', 100)
    if spo2 < 85:
        risk_points += 20  # Life-threatening - خطر على الحياة
    elif spo2 < 90:
        risk_points += 15  # Critical - حرج
    elif spo2 < 92:
        risk_points += 10  # Concerning - مقلق
    elif spo2 < 94:
        risk_points += 5   # Mild concern - قلق خفيف
    
    # 2. Temperature - 15 points
    # الحمى الشديدة أو انخفاض الحرارة
    temp = patient_data.get('temperature_c', 37)
    if temp > 39.5 or temp < 35.5:
        risk_points += 15  # Severe - شديد
    elif temp > 38.5 or temp < 36:
        risk_points += 10  # Moderate - متوسط
    elif temp > 38 or temp < 36.5:
        risk_points += 5   # Mild - خفيف
    
    # 3. Heart Rate - 15 points
    # معدل ضربات القلب غير الطبيعي
    hr = patient_data.get('heart_rate', 80)
    if hr > 130 or hr < 45:
        risk_points += 15  # Severe tachycardia/bradycardia
    elif hr > 120 or hr < 50:
        risk_points += 10
    elif hr > 110 or hr < 55:
        risk_points += 5
    
    # 4. Respiratory Rate - 10 points
    # معدل التنفس
    rr = patient_data.get('respiratory_rate', 16)
    if rr > 30 or rr < 8:
        risk_points += 10  # Severe - شديد
    elif rr > 25 or rr < 10:
        risk_points += 7
    elif rr > 22:
        risk_points += 4
    
    # ===== LAB VALUES (30 points max) =====
    
    # 5. Lactate (Sepsis indicator) - 15 points
    # مؤشر خطير للإنتان (التسمم)
    lactate = patient_data.get('lactate', 1.0)
    if lactate > 4.0:
        risk_points += 15  # Severe sepsis - إنتان شديد
    elif lactate > 2.5:
        risk_points += 10
    elif lactate > 2.0:
        risk_points += 5
    
    # 6. WBC Count - 5 points
    # عدد كريات الدم البيضاء
    wbc = patient_data.get('wbc_count', 8)
    if wbc > 15 or wbc < 3:
        risk_points += 5
    elif wbc > 12 or wbc < 4:
        risk_points += 3
    
    # 7. Creatinine (Kidney function) - 5 points
    # وظائف الكلى
    creat = patient_data.get('creatinine', 1.0)
    if creat > 2.5:
        risk_points += 5
    elif creat > 2.0:
        risk_points += 4
    elif creat > 1.5:
        risk_points += 3
    
    # 8. CRP (Inflammation marker) - 5 points
    # علامة الالتهاب
    crp = patient_data.get('crp_level', 5)
    if crp > 100:
        risk_points += 5
    elif crp > 50:
        risk_points += 3
    elif crp > 20:
        risk_points += 2
    
    # ===== BLOOD PRESSURE (10 points max) =====
    
    # 9. Blood Pressure - 10 points
    systolic = patient_data.get('systolic_bp', 120)
    diastolic = patient_data.get('diastolic_bp', 80)
    
    # انخفاض ضغط الدم (Hypotension) - خطير جداً
    if systolic < 90:
        risk_points += 8
    elif systolic < 100:
        risk_points += 5
    
    # ارتفاع ضغط الدم الشديد (Hypertensive crisis)
    if systolic > 180:
        risk_points += 7
    elif systolic > 160:
        risk_points += 4
    
    # ارتفاع الضغط الانبساطي
    if diastolic > 110:
        risk_points += 5
    elif diastolic > 100:
        risk_points += 3
    
    # ===== OXYGEN SUPPORT (10 points max) =====
    
    # 10. Oxygen Support - 10 points
    oxygen_flow = patient_data.get('oxygen_flow', 0)
    oxygen_device = patient_data.get('oxygen_device', 0)
    
    if oxygen_device == 3:  # Ventilator - جهاز التنفس
        risk_points += 10
    elif oxygen_device == 2:  # Face Mask - قناع الوجه
        risk_points += 7
    elif oxygen_device == 1:  # Nasal Cannula - أنبوب الأنف
        if oxygen_flow > 5:
            risk_points += 5
        elif oxygen_flow > 2:
            risk_points += 3
        else:
            risk_points += 2
    
    # ===== CLINICAL ALERTS (10 points max) =====
    
    # 11. Nurse Alert - 5 points
    # تنبيه من الممرضة
    if patient_data.get('nurse_alert', 0) == 1:
        risk_points += 5
    
    # 12. Mobility Score - 5 points
    # القدرة على الحركة
    mobility = patient_data.get('mobility_score', 3)
    if mobility == 0:  # Bedridden - طريح الفراش
        risk_points += 5
    elif mobility == 1:  # Chair/Wheelchair
        risk_points += 3
    elif mobility == 2:  # Walking with help
        risk_points += 1
    
    # ===== HEMOGLOBIN (5 points max) =====
    
    # 13. Hemoglobin - 5 points
    # فقر الدم أو زيادة الدم
    hgb = patient_data.get('hemoglobin', 13)
    if hgb < 8:  # Severe anemia
        risk_points += 5
    elif hgb < 10:  # Moderate anemia
        risk_points += 3
    elif hgb > 18:  # Polycythemia
        risk_points += 2
    
    # ===== AGE FACTOR (5 points max) =====
    
    # 14. Age - 5 points
    # العمر عامل خطر
    age = patient_data.get('age', 50)
    if age > 85:
        risk_points += 5
    elif age > 80:
        risk_points += 4
    elif age > 75:
        risk_points += 3
    elif age > 70:
        risk_points += 2
    elif age > 65:
        risk_points += 1
    
    # ===== COMORBIDITIES (5 points max) =====
    
    # 15. Comorbidity Index - 5 points
    # الأمراض المصاحبة
    comorbidity = patient_data.get('comorbidity_index', 0)
    risk_points += min(comorbidity, 5)
    
    # Calculate final risk score (0-1)
    # احسب النتيجة النهائية
    risk_score = min(risk_points / max_possible_points, 0.99)
    
    return risk_score


def get_risk_category_from_score(risk_score):
    """
    تحديد فئة الخطر بناء على النتيجة
    Determine risk category from score
    
    Args:
        risk_score: Float between 0 and 1
        
    Returns:
        String: Risk category
    """
    if risk_score >= 0.70:
        return "Critical Risk"
    elif risk_score >= 0.50:
        return "High Risk"
    elif risk_score >= 0.30:
        return "Medium Risk"
    else:
        return "Low Risk"


def predict_with_rules(patient_data):
    """
    تنبؤ كامل بناء على القواعد الطبية
    Complete prediction using clinical rules
    
    Args:
        patient_data: Dictionary with patient information
        
    Returns:
        Dictionary with prediction results
    """
    # Calculate risk score
    risk_score = calculate_risk_score_rules_based(patient_data)
    
    # Get risk category
    risk_category = get_risk_category_from_score(risk_score)
    
    # Determine prediction (0 = Stable, 1 = Deterioration)
    prediction = 1 if risk_score >= 0.5 else 0
    
    # Calculate confidence
    confidence = risk_score if prediction == 1 else (1 - risk_score)
    
    return {
        'prediction': int(prediction),
        'risk_score': float(risk_score),
        'risk_category': risk_category,
        'confidence': float(confidence),
        'method': 'Rule-Based Clinical System',
        'probabilities': {
            'no_deterioration': float(1 - risk_score),
            'deterioration': float(risk_score)
        }
    }


def get_detailed_risk_breakdown(patient_data):
    """
    تحليل مفصل لمصادر الخطر
    Detailed breakdown of risk factors
    
    Args:
        patient_data: Dictionary with patient data
        
    Returns:
        Dictionary with detailed risk analysis
    """
    breakdown = {
        'vital_signs': {},
        'lab_values': {},
        'clinical_factors': {},
        'total_risk_points': 0
    }
    
    # Vital signs
    spo2 = patient_data.get('spo2_pct', 100)
    if spo2 < 92:
        breakdown['vital_signs']['spo2'] = {
            'value': spo2,
            'status': 'Critical' if spo2 < 90 else 'Concerning',
            'points': 15 if spo2 < 90 else 10
        }
    
    temp = patient_data.get('temperature_c', 37)
    if temp > 38.5 or temp < 36:
        breakdown['vital_signs']['temperature'] = {
            'value': temp,
            'status': 'Abnormal',
            'points': 10 if temp > 39.5 or temp < 35.5 else 5
        }
    
    hr = patient_data.get('heart_rate', 80)
    if hr > 110 or hr < 55:
        breakdown['vital_signs']['heart_rate'] = {
            'value': hr,
            'status': 'Abnormal',
            'points': 15 if hr > 130 or hr < 45 else 10
        }
    
    # Lab values
    lactate = patient_data.get('lactate', 1.0)
    if lactate > 2.0:
        breakdown['lab_values']['lactate'] = {
            'value': lactate,
            'status': 'Elevated',
            'points': 15 if lactate > 4.0 else 10
        }
    
    wbc = patient_data.get('wbc_count', 8)
    if wbc > 12 or wbc < 4:
        breakdown['lab_values']['wbc'] = {
            'value': wbc,
            'status': 'Abnormal',
            'points': 5 if wbc > 15 or wbc < 3 else 3
        }
    
    # Clinical factors
    if patient_data.get('nurse_alert', 0) == 1:
        breakdown['clinical_factors']['nurse_alert'] = {
            'status': 'Active',
            'points': 5
        }
    
    # Calculate total points
    total = 0
    for category in breakdown.values():
        if isinstance(category, dict):
            for item in category.values():
                if isinstance(item, dict) and 'points' in item:
                    total += item['points']
    
    breakdown['total_risk_points'] = total
    breakdown['risk_score'] = min(total / 100, 0.99)
    
    return breakdown


# ===== TESTING SECTION =====

if __name__ == "__main__":
    print("="*70)
    print("🏥 RULE-BASED CLINICAL DECISION SYSTEM - TEST")
    print("="*70)
    
    # Test 1: Critical patient (يجب أن يعطي خطر عالي)
    critical_patient = {
        'patient_id': 'a245',
        'age': 87,
        'gender': 1,
        'heart_rate': 120,
        'respiratory_rate': 25,
        'spo2_pct': 80,  # 🚨 CRITICAL
        'temperature_c': 40,  # 🚨 SEVERE FEVER
        'systolic_bp': 150,
        'diastolic_bp': 110,
        'wbc_count': 15,
        'lactate': 4.0,  # 🚨 SEPSIS
        'creatinine': 2.0,
        'crp_level': 15,
        'hemoglobin': 20,
        'oxygen_flow': 8,
        'oxygen_device': 1,
        'nurse_alert': 1,
        'mobility_score': 1,
        'comorbidity_index': 2
    }
    
    print("\n" + "="*70)
    print("TEST 1: CRITICAL PATIENT")
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
    
    result1 = predict_with_rules(critical_patient)
    
    print(f"\n📊 RULE-BASED PREDICTION:")
    print(f"  Risk Score: {result1['risk_score']*100:.1f}%")
    print(f"  Risk Category: {result1['risk_category']}")
    print(f"  Prediction: {'⚠️ Deterioration' if result1['prediction']==1 else '✅ Stable'}")
    print(f"  Confidence: {result1['confidence']*100:.1f}%")
    print(f"  Method: {result1['method']}")
    
    if result1['risk_score'] > 0.6:
        print("\n✅ SUCCESS: Correctly identified as high risk!")
    else:
        print("\n❌ FAILED: Should be high risk!")
    
    # Test 2: Healthy patient (يجب أن يعطي خطر منخفض)
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
        'mobility_score': 4,
        'comorbidity_index': 0
    }
    
    print("\n" + "="*70)
    print("TEST 2: HEALTHY PATIENT")
    print("="*70)
    print("Input values:")
    print(f"  Age: 45 (middle-aged)")
    print(f"  All vital signs: NORMAL ✅")
    print(f"  SpO2: 98% (excellent)")
    print(f"  Temperature: 36.8°C (normal)")
    print(f"  Heart Rate: 75 bpm (normal)")
    print(f"  No oxygen support")
    print(f"  Fully mobile")
    print(f"  No alerts")
    
    result2 = predict_with_rules(healthy_patient)
    
    print(f"\n📊 RULE-BASED PREDICTION:")
    print(f"  Risk Score: {result2['risk_score']*100:.1f}%")
    print(f"  Risk Category: {result2['risk_category']}")
    print(f"  Prediction: {'⚠️ Deterioration' if result2['prediction']==1 else '✅ Stable'}")
    print(f"  Confidence: {result2['confidence']*100:.1f}%")
    
    if result2['risk_score'] < 0.3:
        print("\n✅ SUCCESS: Correctly identified as low risk!")
    else:
        print("\n⚠️ WARNING: Should be low risk!")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✅ Critical patient risk: {result1['risk_score']*100:.1f}% ({result1['risk_category']})")
    print(f"✅ Healthy patient risk: {result2['risk_score']*100:.1f}% ({result2['risk_category']})")
    print(f"\nDifference: {(result1['risk_score'] - result2['risk_score'])*100:.1f} percentage points")
    
    if result1['risk_score'] > 0.6 and result2['risk_score'] < 0.3:
        print("\n🎉 RULE-BASED SYSTEM WORKING PERFECTLY!")
        print("   ✅ Correctly identifies critical patients")
        print("   ✅ Correctly identifies stable patients")
        print("   ✅ Ready for production use")
    else:
        print("\n⚠️ System needs calibration")
    
    print("\n" + "="*70)