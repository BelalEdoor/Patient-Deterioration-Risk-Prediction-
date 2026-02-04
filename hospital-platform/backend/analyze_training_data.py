"""
تحليل بيانات التدريب وإصلاح النموذج
Training Data Analysis & Model Fix Script
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("🔍 TRAINING DATA ANALYSIS")
print("="*70)

# ابحث عن ملف البيانات
possible_paths = [
    "../data/hospital_deterioration_hourly_panel.csv",
    "data/hospital_deterioration_hourly_panel.csv",
    "/Users/admin/Desktop/Patient_Deterioration_Risk_Prediction_Project/hospital-platform/data/hospital_deterioration_hourly_panel.csv"
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if data_path is None:
    print("\n❌ Cannot find training data CSV file!")
    print("Please specify the correct path to hospital_deterioration_hourly_panel.csv")
    exit(1)

print(f"\n✅ Found data at: {data_path}")

# اقرأ البيانات
df = pd.read_csv(data_path)

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]:,}")
print(f"   Columns: {df.shape[1]}")

print("\n📋 Column Names:")
print(df.columns.tolist())

# تحقق من عمود الهدف
target_col = 'deterioration_next_12h'
if target_col not in df.columns:
    print(f"\n❌ Target column '{target_col}' not found!")
    print("Available columns:", df.columns.tolist())
    exit(1)

print(f"\n🎯 Target Variable: {target_col}")
print(df[target_col].value_counts())
print(f"\nClass Distribution:")
print(f"  Class 0 (No Deterioration): {(df[target_col]==0).sum()} ({(df[target_col]==0).sum()/len(df)*100:.1f}%)")
print(f"  Class 1 (Deterioration): {(df[target_col]==1).sum()} ({(df[target_col]==1).sum()/len(df)*100:.1f}%)")

# احصائيات Features
print("\n📊 Feature Statistics:")
print("-" * 70)

feature_cols = [
    'age', 'gender', 'heart_rate', 'respiratory_rate', 'spo2_pct',
    'temperature_c', 'systolic_bp', 'diastolic_bp', 'wbc_count',
    'lactate', 'creatinine', 'crp_level', 'hemoglobin', 'oxygen_flow',
    'oxygen_device', 'nurse_alert', 'mobility_score', 'comorbidity_index'
]

# تحقق من أمثلة الحالات الخطيرة
print("\n🚨 CRITICAL CASES ANALYSIS:")
print("-" * 70)

deteriorated = df[df[target_col] == 1]
stable = df[df[target_col] == 0]

if len(deteriorated) > 0:
    print(f"\n📈 Deteriorated Patients (Class 1): {len(deteriorated)}")
    print("\nAverage values for deteriorated patients:")
    for col in ['spo2_pct', 'temperature_c', 'heart_rate', 'lactate']:
        if col in df.columns:
            print(f"  {col}: {deteriorated[col].mean():.2f}")
    
    print("\n📉 Stable Patients (Class 0): {len(stable)}")
    print("\nAverage values for stable patients:")
    for col in ['spo2_pct', 'temperature_c', 'heart_rate', 'lactate']:
        if col in df.columns:
            print(f"  {col}: {stable[col].mean():.2f}")
else:
    print("\n❌ NO DETERIORATED CASES FOUND!")
    print("   This explains why the model can't learn!")

# تحقق من القيم الشاذة
print("\n🔍 CHECKING FOR DATA QUALITY ISSUES:")
print("-" * 70)

# SpO2 analysis
if 'spo2_pct' in df.columns:
    critical_spo2 = df[df['spo2_pct'] < 90]
    print(f"\nPatients with SpO2 < 90%: {len(critical_spo2)}")
    if len(critical_spo2) > 0:
        print(f"  How many deteriorated? {critical_spo2[target_col].sum()}")
        print(f"  Percentage: {critical_spo2[target_col].sum()/len(critical_spo2)*100:.1f}%")

# Temperature analysis
if 'temperature_c' in df.columns:
    fever = df[df['temperature_c'] > 38.5]
    print(f"\nPatients with fever (>38.5°C): {len(fever)}")
    if len(fever) > 0:
        print(f"  How many deteriorated? {fever[target_col].sum()}")
        print(f"  Percentage: {fever[target_col].sum()/len(fever)*100:.1f}%")

# Lactate analysis  
if 'lactate' in df.columns:
    high_lactate = df[df['lactate'] > 2.0]
    print(f"\nPatients with high lactate (>2.0): {len(high_lactate)}")
    if len(high_lactate) > 0:
        print(f"  How many deteriorated? {high_lactate[target_col].sum()}")
        print(f"  Percentage: {high_lactate[target_col].sum()/len(high_lactate)*100:.1f}%")

print("\n" + "="*70)
print("🔧 DIAGNOSIS & RECOMMENDATIONS")
print("="*70)

# Check for common issues
issues = []

# Issue 1: Extreme class imbalance
imbalance_ratio = (df[target_col]==0).sum() / max((df[target_col]==1).sum(), 1)
if imbalance_ratio > 50:
    issues.append(f"SEVERE class imbalance (ratio: {imbalance_ratio:.1f}:1)")

# Issue 2: No deterioration cases
if (df[target_col]==1).sum() == 0:
    issues.append("NO deterioration cases in dataset!")

# Issue 3: Labels might be inverted
if len(deteriorated) > 0 and len(stable) > 0:
    if deteriorated['spo2_pct'].mean() > stable['spo2_pct'].mean():
        issues.append("⚠️ LABELS MIGHT BE INVERTED! Deteriorated patients have BETTER vitals")

if issues:
    print("\n🚨 ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 RECOMMENDED FIXES:")
    print("  1. If labels are inverted, flip them: df['deterioration_next_12h'] = 1 - df['deterioration_next_12h']")
    print("  2. Use SMOTE or class weights to handle imbalance")
    print("  3. Verify data collection process")
    print("  4. Consider using a rule-based system as fallback")
else:
    print("\n✅ Data appears to be correctly structured")

print("\n" + "="*70)

# Show sample of deteriorated cases
if len(deteriorated) > 0:
    print("\n📋 SAMPLE OF DETERIORATED CASES:")
    print(deteriorated[['spo2_pct', 'temperature_c', 'heart_rate', 'lactate', target_col]].head(10))

print("\n📋 SAMPLE OF STABLE CASES:")
print(stable[['spo2_pct', 'temperature_c', 'heart_rate', 'lactate', target_col]].head(10))

print("\n" + "="*70)