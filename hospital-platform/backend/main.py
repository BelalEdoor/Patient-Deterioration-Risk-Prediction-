"""
FastAPI Backend for Hospital Patient Deterioration Prediction System
Backend API لنظام التنبؤ بتدهور حالة المرضى

This file contains API endpoints with AUTO-TRAINING on first run.
يقوم بالتدريب التلقائي عند أول تشغيل
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime      
import os
import uvicorn

# Import ML model
from ML_Model import PatientRiskModel, get_clinical_recommendations


# ===== Configuration =====

# مسار ملف البيانات التدريبية
# Path to training data CSV file
TRAINING_DATA_PATH = "../data/hospital_deterioration_hourly_panel.csv"  

# معاملات التدريب
# Training parameters
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2
}


# ===== FastAPI App Initialization =====

app = FastAPI(
    title="Hospital Patient Deterioration Prediction API",
    description="REST API for predicting patient deterioration risk with auto-training",
    version="2.0.0"
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
    """
    تدريب النموذج تلقائياً عند أول تشغيل
    Auto-train model on first run if not already trained
    
    Returns:
        bool: True if training was performed, False if model already exists
    """
    # تحقق إذا كان النموذج موجود مسبقاً
    if ml_model.is_trained():
        print("✅ Model already trained and loaded")
        return False
    
    print("\n" + "="*70)
    print("🎓 AUTO-TRAINING MODE - First Run Detected")
    print("="*70)
    print(f"📁 Looking for training data at: {TRAINING_DATA_PATH}")
    
    # تحقق من وجود ملف البيانات
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"\n⚠️  WARNING: Training data not found at {TRAINING_DATA_PATH}")
        print("   Please ensure sample_data.csv exists in the parent directory")
        print("   Or update TRAINING_DATA_PATH in main.py")
        print("\n   Model will NOT be trained automatically.")
        print("   You can train later via POST /train endpoint")
        print("="*70 + "\n")
        return False
    
    print(f"✅ Training data found!")
    print(f"\n🚀 Starting automatic training...")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Validation split: {TRAINING_CONFIG['validation_split']}")
    print("\n" + "="*70 + "\n")
    
    try:
        # بدء التدريب
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
        print(f"   ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ AUTO-TRAINING FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        print("\nPlease check:")
        print("  1. Training data file exists and is valid CSV")
        print("  2. All required columns are present")
        print("  3. TensorFlow is installed correctly")
        print("\nYou can try manual training via POST /train endpoint")
        print("="*70 + "\n")
        return False


# ===== Pydantic Models (Data Validation) =====

class PatientData(BaseModel):
    """Patient data input model"""
    patient_id: str
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    gender: int = Field(..., ge=0, le=1, description="0: Female, 1: Male")
    heart_rate: float = Field(..., ge=30, le=200, description="Heart rate (bpm)")
    respiratory_rate: float = Field(..., ge=5, le=60, description="Respiratory rate (breaths/min)")
    spo2_pct: float = Field(..., ge=70, le=100, description="Oxygen saturation (%)")
    temperature_c: float = Field(..., ge=35, le=42, description="Body temperature (°C)")
    systolic_bp: float = Field(..., ge=60, le=250, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=30, le=150, description="Diastolic blood pressure (mmHg)")
    wbc_count: float = Field(..., ge=1, le=50, description="White blood cell count")
    lactate: float = Field(..., ge=0, le=20, description="Lactate level (mmol/L)")
    creatinine: float = Field(..., ge=0.1, le=15, description="Creatinine level (mg/dL)")
    crp_level: float = Field(..., ge=0, le=500, description="C-reactive protein (mg/L)")
    hemoglobin: float = Field(..., ge=4, le=20, description="Hemoglobin level (g/dL)")
    oxygen_flow: float = Field(..., ge=0, le=15, description="Oxygen flow rate (L/min)")
    oxygen_device: int = Field(..., ge=0, le=3, description="0: None, 1: Nasal, 2: Mask, 3: Ventilator")
    nurse_alert: int = Field(..., ge=0, le=1, description="0: No, 1: Yes")
    mobility_score: int = Field(..., ge=0, le=4, description="0: Bedridden, 4: Fully mobile")
    comorbidity_index: int = Field(..., ge=0, le=10, description="Number of comorbidities")


class PredictionResponse(BaseModel):
    """Prediction result output model"""
    patient_id: str
    risk_score: float
    risk_category: str
    prediction: int
    confidence: float
    timestamp: str
    recommendations: List[str]


class TrainingStatus(BaseModel):
    """Training status output model"""
    status: str
    message: str
    metrics: Optional[Dict] = None
    timestamp: str


class ModelInfo(BaseModel):
    """Model information output model"""
    trained: bool
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    features: Optional[List[str]] = None
    model_path: Optional[str] = None
    last_updated: Optional[str] = None


# ===== API Endpoints =====

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint
    
    Returns API status and basic information
    """
    return {
        "status": "online",
        "service": "Hospital Patient Deterioration Prediction API",
        "model_loaded": ml_model.is_trained(),
        "version": "2.0.0",
        "auto_training": "enabled",
        "docs": "/docs"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_deterioration(patient_data: PatientData):
    """
    Predict patient deterioration risk
    
    Takes patient vital signs and lab results, returns risk assessment.
    
    Args:
        patient_data: Patient information with all required features
        
    Returns:
        PredictionResponse with risk score, category, and recommendations
    """
    # Check if model is trained
    if not ml_model.is_trained():
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please ensure training data exists and restart the server."
        )
    
    try:
        # Convert Pydantic model to dict
        patient_dict = patient_data.dict()
        
        # Get prediction from ML model
        result = ml_model.predict(patient_dict)
        
        # Generate clinical recommendations
        recommendations = get_clinical_recommendations(patient_dict, result['risk_score'])
        
        # Prepare response
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
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(patients: List[PatientData]):
    """
    Predict deterioration for multiple patients
    
    Batch prediction endpoint for processing multiple patients at once.
    
    Args:
        patients: List of patient data
        
    Returns:
        Dictionary with list of predictions and total count
    """
    if not ml_model.is_trained():
        raise HTTPException(
            status_code=503, 
            detail="Model not trained"
        )
    
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


@app.post("/train", response_model=TrainingStatus, tags=["Model Management"])
async def train_model(file: UploadFile = File(...)):
    """
    Manually retrain the model with new data
    
    Upload a CSV file with training data. The model will be retrained.
    Note: Auto-training happens on first run, this is for retraining.
    
    Args:
        file: CSV file with patient data and outcomes
        
    Returns:
        TrainingStatus with success/failure and performance metrics
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    try:
        # Save uploaded file temporarily
        temp_path = "temp_training_data.csv"
        contents = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Train the model
        print(f"📁 Manual retraining with file: {file.filename}")
        metrics = ml_model.train(
            data_path=temp_path,
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            validation_split=TRAINING_CONFIG['validation_split']
        )
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Prepare response
        response = TrainingStatus(
            status="success",
            message="Model retrained successfully",
            metrics=metrics,
            timestamp=datetime.now().isoformat()
        )
        
        print("✅ Manual retraining completed successfully!")
        return response
    
    except ValueError as e:
        raise HTTPException(
            status_code=400, 
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Training error: {str(e)}"
        )


@app.get("/model/status", tags=["Model Management"])
async def get_model_status():
    """
    Get current model status and information
    
    Returns information about the loaded model including features and paths.
    """
    if not ml_model.is_trained():
        return {
            "trained": False,
            "message": "Model not trained yet. Check if training data exists and restart server."
        }
    
    # Get model information
    info = ml_model.get_model_info()
    
    # Get last modified time of model file
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
    """
    Get feature importance from the trained model
    
    Returns ranking of which patient features are most important for predictions.
    Note: For Neural Networks, this returns equal importance as they don't have
    built-in feature importance like tree-based models.
    """
    if not ml_model.is_trained():
        raise HTTPException(
            status_code=404,
            detail="Model not trained yet"
        )
    
    try:
        importance = ml_model.get_feature_importance()
        return {
            "feature_importance": importance,
            "note": "Neural Networks don't have built-in feature importance. Consider using SHAP or permutation importance for detailed analysis."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting feature importance: {str(e)}"
        )


@app.delete("/model", tags=["Model Management"])
async def delete_model():
    """
    Delete the current trained model
    
    Removes saved model files. Next server restart will retrain automatically.
    """
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
        
        # Reset model instance
        ml_model.model = None
        ml_model.scaler = None
        ml_model.feature_names = []
        
        if deleted_files:
            return {
                "status": "success",
                "message": "Model deleted successfully. Restart server to retrain automatically.",
                "deleted_files": deleted_files
            }
        else:
            return {
                "status": "info",
                "message": "No model files found to delete"
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting model: {str(e)}"
        )


# ===== Startup Event =====

@app.on_event("startup")
async def startup_event():
    """
    Initialize on server startup with AUTO-TRAINING
    
    - Checks if model exists
    - If not, automatically trains using TRAINING_DATA_PATH
    - If yes, loads existing model
    """
    print("\n" + "="*70)
    print("🏥 Hospital Patient Risk Prediction API v2.0")
    print("   WITH AUTO-TRAINING ON FIRST RUN")
    print("="*70)
    
    # تنفيذ التدريب التلقائي
    training_performed = auto_train_model()
    
    if ml_model.is_trained():
        info = ml_model.get_model_info()
        print("\n📊 Model Status:")
        print(f"   ✅ Model Type: {info['model_type']}")
        print(f"   ✅ Architecture: {info['architecture']}")
        print(f"   ✅ Total Parameters: {info['total_params']:,}")
        print(f"   ✅ Features: {info['n_features']}")
        
        if training_performed:
            print(f"\n   🎓 Status: NEWLY TRAINED (First Run)")
        else:
            print(f"\n   📁 Status: LOADED FROM DISK")
    else:
        print("\n⚠️  Model Status: NOT TRAINED")
        print("   Predictions will NOT be available until model is trained")
        print(f"   Place training data at: {TRAINING_DATA_PATH}")
        print("   Or use POST /train endpoint to upload data")
    
    print("\n" + "="*70)
    print(f"📡 API running at: http://localhost:8000")
    print(f"📖 Documentation: http://localhost:8000/docs")
    print(f"🎯 Predictions: http://localhost:8000/predict")
    print("="*70 + "\n")


# ===== Run Server =====

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )