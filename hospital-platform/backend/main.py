"""
FastAPI Backend for Hospital Patient Deterioration Prediction System
Backend API لنظام التنبؤ بتدهور حالة المرضى

Enhanced version with CSV upload support for multi-hour patient readings
نسخة محسّنة تدعم رفع ملفات CSV لقراءات متعددة للمريض
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime      
import os
import uvicorn
import pandas as pd
import numpy as np
import io

# Import ML model
from ML_Model import PatientRiskModel, get_clinical_recommendations


# ===== Configuration =====

TRAINING_DATA_PATH = "../data/hospital_deterioration_hourly_panel.csv"  

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2
}


# ===== FastAPI App Initialization =====

app = FastAPI(
    title="Hospital Patient Deterioration Prediction API",
    description="REST API for predicting patient deterioration risk with CSV upload support",
    version="3.0.0"
)


# ===== CORS Configuration =====

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Global Model Instance =====

ml_model = PatientRiskModel(model_dir="models")


# ===== Auto-Training Function =====

def auto_train_model():
    """تدريب النموذج تلقائياً عند أول تشغيل"""
    if ml_model.is_trained():
        print("✅ Model already trained and loaded")
        return False
    
    print("\n" + "="*70)
    print("🎓 AUTO-TRAINING MODE - First Run Detected")
    print("="*70)
    print(f"📁 Looking for training data at: {TRAINING_DATA_PATH}")
    
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"\n⚠️  WARNING: Training data not found at {TRAINING_DATA_PATH}")
        print("   Model will NOT be trained automatically.")
        print("="*70 + "\n")
        return False
    
    print(f"✅ Training data found!")
    print(f"\n🚀 Starting automatic training...")
    print("\n" + "="*70 + "\n")
    
    try:
        metrics = ml_model.train(
            data_path=TRAINING_DATA_PATH,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_split=TRAINING_CONFIG['validation_split']
        )
        
        print("\n" + "="*70)
        print("✅ AUTO-TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📊 Final Metrics:")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']*100:.2f}%")
        print(f"   Recall:    {metrics['recall']*100:.2f}% ⭐")
        print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ AUTO-TRAINING FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("="*70 + "\n")
        return False


# ===== Pydantic Models =====

class PatientData(BaseModel):
    """Patient data input model"""
    patient_id: str
    age: float = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=0, le=1)
    heart_rate: float = Field(..., ge=30, le=200)
    respiratory_rate: float = Field(..., ge=5, le=60)
    spo2_pct: float = Field(..., ge=70, le=100)
    temperature_c: float = Field(..., ge=35, le=42)
    systolic_bp: float = Field(..., ge=60, le=250)
    diastolic_bp: float = Field(..., ge=30, le=150)
    wbc_count: float = Field(..., ge=1, le=50)
    lactate: float = Field(..., ge=0, le=20)
    creatinine: float = Field(..., ge=0.1, le=15)
    crp_level: float = Field(..., ge=0, le=500)
    hemoglobin: float = Field(..., ge=4, le=20)
    oxygen_flow: float = Field(..., ge=0, le=15)
    oxygen_device: int = Field(..., ge=0, le=3)
    nurse_alert: int = Field(..., ge=0, le=1)
    mobility_score: int = Field(..., ge=0, le=4)
    comorbidity_index: int = Field(..., ge=0, le=10)


class PredictionResponse(BaseModel):
    """Prediction result output model"""
    patient_id: str
    risk_score: float
    risk_category: str
    prediction: int
    confidence: float
    timestamp: str
    recommendations: List[str]


class HourlyReading(BaseModel):
    """Single hourly reading"""
    hour: int
    risk_score: float
    risk_category: str
    prediction: int
    spo2: float
    heart_rate: float
    temperature: float
    lactate: float


class CSVPredictionResponse(BaseModel):
    """Response for CSV file prediction"""
    patient_id: str
    total_hours: int
    deterioration_detected: bool
    deterioration_hour: Optional[int]
    final_risk_score: float
    final_risk_category: str
    hourly_readings: List[HourlyReading]
    summary: Dict
    recommendations: List[str]


# ===== Helper Functions =====

def process_csv_file(file_contents: bytes) -> pd.DataFrame:
    """
    معالجة ملف CSV وتحويله إلى DataFrame
    Process CSV file and convert to DataFrame
    """
    try:
        # Read CSV from bytes
        df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
        
        # Validate required columns
        required_columns = [
            'patient_id', 'age', 'gender', 'heart_rate', 'respiratory_rate',
            'spo2_pct', 'temperature_c', 'systolic_bp', 'diastolic_bp',
            'wbc_count', 'lactate', 'creatinine', 'crp_level', 'hemoglobin',
            'oxygen_flow', 'oxygen_device', 'nurse_alert', 'mobility_score',
            'comorbidity_index'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")


def analyze_patient_trajectory(df: pd.DataFrame) -> Dict:
    """
    تحليل مسار المريض عبر الساعات
    Analyze patient trajectory over hours
    """
    if not ml_model.is_trained():
        raise HTTPException(status_code=503, detail="Model not trained")
    
    patient_id = df['patient_id'].iloc[0]
    hourly_predictions = []
    deterioration_detected = False
    deterioration_hour = None
    
    # Process each hour
    for idx, row in df.iterrows():
        # Extract patient data
        patient_dict = row.to_dict()
        
        # Get prediction
        result = ml_model.predict(patient_dict)
        
        # Store hourly reading
        hourly_reading = {
            'hour': int(row.get('hour', idx)),
            'risk_score': float(result['risk_score']),
            'risk_category': result['risk_category'],
            'prediction': int(result['prediction']),
            'spo2': float(row['spo2_pct']),
            'heart_rate': float(row['heart_rate']),
            'temperature': float(row['temperature_c']),
            'lactate': float(row['lactate'])
        }
        
        hourly_predictions.append(hourly_reading)
        
        # Check for deterioration
        if result['prediction'] == 1 and not deterioration_detected:
            deterioration_detected = True
            deterioration_hour = hourly_reading['hour']
    
    # Get final reading
    final_reading = hourly_predictions[-1]
    
    # Generate recommendations based on final state
    final_patient_dict = df.iloc[-1].to_dict()
    recommendations = get_clinical_recommendations(
        final_patient_dict, 
        final_reading['risk_score']
    )
    
    # Calculate summary statistics
    risk_scores = [r['risk_score'] for r in hourly_predictions]
    summary = {
        'min_risk': float(np.min(risk_scores)),
        'max_risk': float(np.max(risk_scores)),
        'avg_risk': float(np.mean(risk_scores)),
        'risk_trend': 'increasing' if risk_scores[-1] > risk_scores[0] else 'stable/decreasing',
        'hours_high_risk': sum(1 for r in hourly_predictions if r['risk_score'] >= 0.7),
        'hours_medium_risk': sum(1 for r in hourly_predictions if 0.5 <= r['risk_score'] < 0.7),
        'hours_low_risk': sum(1 for r in hourly_predictions if r['risk_score'] < 0.5)
    }
    
    return {
        'patient_id': patient_id,
        'total_hours': len(hourly_predictions),
        'deterioration_detected': deterioration_detected,
        'deterioration_hour': deterioration_hour,
        'final_risk_score': final_reading['risk_score'],
        'final_risk_category': final_reading['risk_category'],
        'hourly_readings': hourly_predictions,
        'summary': summary,
        'recommendations': recommendations
    }


# ===== API Endpoints =====

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Hospital Patient Deterioration Prediction API",
        "model_loaded": ml_model.is_trained(),
        "version": "3.0.0",
        "features": ["single_prediction", "batch_prediction", "csv_upload"],
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_deterioration(patient_data: PatientData):
    """
    Predict patient deterioration risk (single reading)
    
    التنبؤ بخطر تدهور المريض (قراءة واحدة)
    """
    if not ml_model.is_trained():
        raise HTTPException(
            status_code=503,
            detail="Model not trained"
        )
    
    try:
        patient_dict = patient_data.dict()
        result = ml_model.predict(patient_dict)
        recommendations = get_clinical_recommendations(patient_dict, result['risk_score'])
        
        response = PredictionResponse(
            patient_id=patient_data.patient_id,
            risk_score=round(result['risk_score'], 3),
            risk_category=result['risk_category'],
            prediction=result['prediction'],
            confidence=round(result['confidence'], 3),
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv", response_model=CSVPredictionResponse, tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict patient deterioration from CSV file with multiple hourly readings
    
    التنبؤ بتدهور المريض من ملف CSV يحتوي على قراءات متعددة لساعات مختلفة
    
    CSV Format:
    - Must contain columns: hour, patient_id, age, gender, heart_rate, respiratory_rate, 
      spo2_pct, temperature_c, systolic_bp, diastolic_bp, wbc_count, lactate, 
      creatinine, crp_level, hemoglobin, oxygen_flow, oxygen_device, 
      nurse_alert, mobility_score, comorbidity_index
    - Each row represents one hour of patient data
    - Sorted by hour (ascending)
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported. الرجاء رفع ملف CSV فقط"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Process CSV
        df = process_csv_file(contents)
        
        print(f"\n📊 Processing CSV file: {file.filename}")
        print(f"   Patient ID: {df['patient_id'].iloc[0]}")
        print(f"   Total readings: {len(df)}")
        
        # Analyze patient trajectory
        result = analyze_patient_trajectory(df)
        
        print(f"   Deterioration detected: {result['deterioration_detected']}")
        if result['deterioration_detected']:
            print(f"   Deterioration at hour: {result['deterioration_hour']}")
        print(f"   Final risk score: {result['final_risk_score']*100:.1f}%")
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(patients: List[PatientData]):
    """Batch prediction for multiple patients"""
    if not ml_model.is_trained():
        raise HTTPException(status_code=503, detail="Model not trained")
    
    results = []
    for patient in patients:
        try:
            patient_dict = patient.dict()
            result = ml_model.predict(patient_dict)
            
            results.append({
                "patient_id": patient.patient_id,
                "risk_score": round(result['risk_score'], 3),
                "risk_category": result['risk_category'],
                "prediction": result['prediction'],
                "confidence": round(result['confidence'], 3)
            })
        except Exception as e:
            results.append({
                "patient_id": patient.patient_id,
                "error": str(e)
            })
    
    return {
        "predictions": results, 
        "total": len(results)
    }


@app.post("/train", tags=["Model Management"])
async def train_model(file: UploadFile = File(...)):
    """Manually retrain the model with new data"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        temp_path = "temp_training_data.csv"
        contents = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        print(f"📁 Manual retraining with file: {file.filename}")
        metrics = ml_model.train(
            data_path=temp_path,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_split=TRAINING_CONFIG['validation_split']
        )
        
        os.remove(temp_path)
        
        print("✅ Manual retraining completed successfully!")
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")


@app.get("/model/status", tags=["Model Management"])
async def get_model_status():
    """Get current model status and information"""
    if not ml_model.is_trained():
        return {
            "trained": False,
            "message": "Model not trained yet"
        }
    
    info = ml_model.get_model_info()
    
    last_updated = None
    if os.path.exists(ml_model.model_path):
        last_updated = datetime.fromtimestamp(
            os.path.getmtime(ml_model.model_path)
        ).isoformat()
    
    return {
        "trained": info['trained'],
        "model_type": info['model_type'],
        "features_count": info['n_features'],
        "features": info['features'],
        "model_path": info['model_path'],
        "last_updated": last_updated
    }


@app.get("/model/importance", tags=["Model Management"])
async def get_feature_importance():
    """Get feature importance from the trained model"""
    if not ml_model.is_trained():
        raise HTTPException(status_code=404, detail="Model not trained yet")
    
    try:
        importance = ml_model.get_feature_importance()
        return {
            "feature_importance": importance,
            "note": "Neural Networks don't have built-in feature importance"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/model", tags=["Model Management"])
async def delete_model():
    """Delete the current trained model"""
    try:
        deleted_files = []
        
        if os.path.exists(ml_model.model_path):
            os.remove(ml_model.model_path)
            deleted_files.append("patient_nn_model.h5")
        
        if os.path.exists(ml_model.scaler_path):
            os.remove(ml_model.scaler_path)
            deleted_files.append("scaler.pkl")
        
        if os.path.exists(ml_model.feature_names_path):
            os.remove(ml_model.feature_names_path)
            deleted_files.append("feature_names.json")
        
        ml_model.model = None
        ml_model.scaler = None
        ml_model.feature_names = []
        
        if deleted_files:
            return {
                "status": "success",
                "message": "Model deleted successfully",
                "deleted_files": deleted_files
            }
        else:
            return {
                "status": "info",
                "message": "No model files found to delete"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Initialize on server startup with AUTO-TRAINING"""
    print("\n" + "="*70)
    print("🏥 Hospital Patient Risk Prediction API v3.0")
    print("   WITH CSV UPLOAD SUPPORT")
    print("="*70)
    
    training_performed = auto_train_model()
    
    if ml_model.is_trained():
        info = ml_model.get_model_info()
        print("\n📊 Model Status:")
        print(f"   ✅ Model Type: {info['model_type']}")
        print(f"   ✅ Architecture: {info['architecture']}")
        print(f"   ✅ Features: {info['n_features']}")
        
        if training_performed:
            print(f"\n   🎓 Status: NEWLY TRAINED (First Run)")
        else:
            print(f"\n   📁 Status: LOADED FROM DISK")
    else:
        print("\n⚠️  Model Status: NOT TRAINED")
        print(f"   Place training data at: {TRAINING_DATA_PATH}")
    
    print("\n" + "="*70)
    print(f"📡 API running at: http://localhost:8000")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print(f"🎯 Single Prediction: POST /predict")
    print(f"📊 CSV Upload: POST /predict/csv")
    print("="*70 + "\n")


# ===== Run Server =====

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )