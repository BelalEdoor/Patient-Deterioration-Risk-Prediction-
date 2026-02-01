"""
Neural Network Model Module
نموذج الشبكة العصبية للتنبؤ بتدهور حالة المرضى
Patient Deterioration Risk Prediction - Neural Network Model
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report, 
    confusion_matrix, 
    roc_auc_score
)


# ===== Encoding Maps (Global) =====
# خرائط التحويل للأعمدة النصية إلى أرقام

GENDER_MAP = {
    'M': 1, 'Male': 1, 'male': 1, 'm': 1,
    'F': 0, 'Female': 0, 'female': 0, 'f': 0
}

OXYGEN_DEVICE_MAP = {
    'none': 0, 'None': 0, 'NONE': 0,
    'nasal': 1, 'Nasal': 1, 'NASAL': 1, 'nasal_cannula': 1,
    'mask': 2, 'Mask': 2, 'MASK': 2, 'face_mask': 2,
    'ventilator': 3, 'Ventilator': 3, 'VENTILATOR': 3, 'vent': 3
}


def encode_categorical_columns(df):
    """
    تحويل الأعمدة النصية إلى أرقام
    Encode categorical text columns to numeric values
    
    Args:
        df: DataFrame with possible text columns
        
    Returns:
        DataFrame with encoded numeric columns
    """
    df = df.copy()

    # Encode gender
    if 'gender' in df.columns:
        if df['gender'].dtype == object:
            df['gender'] = df['gender'].map(GENDER_MAP)
            # إذا فيه قيم مش في الخريطة، نحولها لـ NaN ونملأها بالقيمة الوسطى
            df['gender'] = df['gender'].fillna(df['gender'].median())

    # Encode oxygen_device
    if 'oxygen_device' in df.columns:
        if df['oxygen_device'].dtype == object:
            df['oxygen_device'] = df['oxygen_device'].map(OXYGEN_DEVICE_MAP)
            df['oxygen_device'] = df['oxygen_device'].fillna(df['oxygen_device'].median())

    return df


def encode_single_patient(patient_dict):
    """
    تحويل قيم مريض واحد من نص إلى أرقام
    Encode a single patient dictionary's categorical values
    
    Args:
        patient_dict: Dictionary with patient data
        
    Returns:
        Dictionary with encoded values
    """
    patient_dict = patient_dict.copy()

    # Encode gender
    if isinstance(patient_dict.get('gender'), str):
        patient_dict['gender'] = GENDER_MAP.get(patient_dict['gender'], 0)

    # Encode oxygen_device
    if isinstance(patient_dict.get('oxygen_device'), str):
        patient_dict['oxygen_device'] = OXYGEN_DEVICE_MAP.get(patient_dict['oxygen_device'], 0)

    return patient_dict


class PatientRiskModel:
    """
    نموذج الشبكة العصبية للتنبؤ بخطر تدهور المرضى
    Neural Network for Patient Deterioration Risk Prediction
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the Neural Network model
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "patient_nn_model.h5")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.feature_names_path = os.path.join(model_dir, "feature_names.json")
        
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        # Create models directory if not exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def get_feature_columns(self):
        """
        Get the list of feature column names
        
        Returns:
            List of feature names in correct order
        """
        return [
            'age', 'gender', 'heart_rate', 'respiratory_rate', 'spo2_pct',
            'temperature_c', 'systolic_bp', 'diastolic_bp', 'wbc_count',
            'lactate', 'creatinine', 'crp_level', 'hemoglobin', 'oxygen_flow',
            'oxygen_device', 'nurse_alert', 'mobility_score', 'comorbidity_index'
        ]
    
    def _build_neural_network(self, input_dim, class_weight_dict=None):
        """
        Build and compile the Neural Network architecture
        
        Args:
            input_dim: Number of input features
            class_weight_dict: Dictionary for class balancing
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 1
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layer 2
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Hidden layer 3
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def train(self, data_path=None, df=None, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the Neural Network model
        
        Args:
            data_path: Path to CSV file (optional)
            df: Pandas DataFrame (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            
        Returns:
            Dictionary with training metrics
        """
        # Load data
        if data_path:
            df = pd.read_csv(data_path)
        elif df is None:
            raise ValueError("Either data_path or df must be provided")
        
        print(f"📊 Dataset loaded: {df.shape}")
        
        # ===== الإصلاح: تحويل الأعمدة النصية قبل المعالجة =====
        print("🔄 Encoding categorical columns (gender, oxygen_device)...")
        df = encode_categorical_columns(df)
        print("✅ Categorical encoding complete")
        # ============================================================
        
        # Define features and target
        feature_cols = self.get_feature_columns()
        target_col = 'deterioration_next_12h'
        
        # Check if all required columns exist
        missing_cols = set(feature_cols + [target_col]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Prepare data
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # تحقق إن كل الأعمدة رقمية بعد الـ encoding
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValueError(f"These columns are still non-numeric after encoding: {non_numeric}")
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📈 Train set: {len(X_train)} samples")
        print(f"📉 Test set: {len(X_test)} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights
        class_counts = y_train.value_counts()
        total = len(y_train)
        class_weight_dict = {
            0: total / (2 * class_counts[0]),
            1: total / (2 * class_counts[1])
        }
        
        print(f"⚖️  Class weights: {class_weight_dict}")
        print(f"   Class 0 (No deterioration): {class_counts[0]} samples")
        print(f"   Class 1 (Deterioration): {class_counts[1]} samples")
        
        # Build Neural Network
        print("🧠 Building Neural Network...")
        self.model = self._build_neural_network(
            input_dim=X_train_scaled.shape[1],
            class_weight_dict=class_weight_dict
        )
        
        # Print model summary
        print("\n" + "="*60)
        print("🏗️  NEURAL NETWORK ARCHITECTURE")
        print("="*60)
        self.model.summary()
        print("="*60 + "\n")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_recall',
            mode='max',
            patience=20,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        print("🚀 Training Neural Network...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Validation split: {validation_split}")
        print("\n")
        
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✅ Model training completed!")
        
        # Evaluate model
        print("📊 Evaluating model on test set...")
        y_pred_proba = self.model.predict(X_test_scaled).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "epochs_trained": len(history.history['loss'])
        }
        
        print("\n" + "="*60)
        print("📊 NEURAL NETWORK PERFORMANCE METRICS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}% (Sensitivity) ⭐")
        print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
        print(f"ROC-AUC:   {metrics['roc_auc']*100:.2f}%")
        print(f"\nEpochs trained: {metrics['epochs_trained']}")
        print("="*60)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0][0]}")
        print(f"   False Positives: {cm[0][1]}")
        print(f"   False Negatives: {cm[1][0]} ⚠️  (Most critical to minimize)")
        print(f"   True Positives:  {cm[1][1]}")
        print("="*60 + "\n")
        
        # Save feature names
        self.feature_names = feature_cols
        
        # Save model artifacts
        self.save_model()
        
        return metrics
    
    def predict(self, patient_data):
        """
        Predict deterioration risk for a single patient
        
        Args:
            patient_data: Dictionary or array-like with patient features
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        if isinstance(patient_data, dict):
            # ===== الإصلاح: تحويل القيم النصية قبل التنبؤ =====
            patient_data = encode_single_patient(patient_data)
            # =====================================================
            features = self._prepare_features_from_dict(patient_data)
        else:
            features = np.array(patient_data).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get prediction probability
        risk_score = float(self.model.predict(features_scaled, verbose=0).flatten()[0])
        prediction = 1 if risk_score >= 0.5 else 0
        
        # Get risk category
        risk_category = self._get_risk_category(risk_score)
        
        # Confidence is the probability of the predicted class
        confidence = risk_score if prediction == 1 else (1 - risk_score)
        
        return {
            'prediction': int(prediction),
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'confidence': float(confidence),
            'probabilities': {
                'no_deterioration': float(1 - risk_score),
                'deterioration': float(risk_score)
            }
        }
    
    def predict_batch(self, patients_data):
        """
        Predict deterioration risk for multiple patients
        
        Args:
            patients_data: List of dictionaries or DataFrame
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        results = []
        
        if isinstance(patients_data, pd.DataFrame):
            # ===== الإصلاح: تحويل الـ DataFrame قبل التنبؤ =====
            patients_data = encode_categorical_columns(patients_data)
            # =====================================================
            X = patients_data[self.feature_names]
            X_scaled = self.scaler.transform(X)
            predictions_proba = self.model.predict(X_scaled, verbose=0).flatten()
            predictions = (predictions_proba >= 0.5).astype(int)
            
            for i in range(len(X)):
                risk_score = float(predictions_proba[i])
                prediction = int(predictions[i])
                confidence = risk_score if prediction == 1 else (1 - risk_score)
                
                results.append({
                    'prediction': prediction,
                    'risk_score': risk_score,
                    'risk_category': self._get_risk_category(risk_score),
                    'confidence': float(confidence)
                })
        else:
            for patient in patients_data:
                results.append(self.predict(patient))
        
        return results
    
    def _prepare_features_from_dict(self, patient_dict):
        """
        Convert patient dictionary to feature array
        
        Args:
            patient_dict: Dictionary with patient data (already encoded)
            
        Returns:
            Numpy array with features in correct order
        """
        features = [
            patient_dict['age'],
            patient_dict['gender'],
            patient_dict['heart_rate'],
            patient_dict['respiratory_rate'],
            patient_dict['spo2_pct'],
            patient_dict['temperature_c'],
            patient_dict['systolic_bp'],
            patient_dict['diastolic_bp'],
            patient_dict['wbc_count'],
            patient_dict['lactate'],
            patient_dict['creatinine'],
            patient_dict['crp_level'],
            patient_dict['hemoglobin'],
            patient_dict['oxygen_flow'],
            patient_dict['oxygen_device'],
            patient_dict['nurse_alert'],
            patient_dict['mobility_score'],
            patient_dict['comorbidity_index']
        ]
        return np.array(features, dtype=np.float64).reshape(1, -1)
    
    def _get_risk_category(self, risk_score):
        """
        Determine risk category based on risk score
        
        Args:
            risk_score: Float between 0 and 1
            
        Returns:
            String: 'High Risk', 'Medium Risk', or 'Low Risk'
        """
        if risk_score >= 0.7:
            return "High Risk"
        elif risk_score >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    def get_feature_importance(self):
        """
        Get approximate feature importance using permutation
        Note: Neural Networks don't have built-in feature importance like tree models
        
        Returns:
            Dictionary with feature names and approximate importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # For Neural Networks, we return equal importance or use permutation importance
        # This is a simplified version - for real importance, use permutation_importance
        importances = np.ones(len(self.feature_names)) / len(self.feature_names)
        feature_importance = dict(zip(self.feature_names, importances))
        
        return feature_importance
    
    def save_model(self):
        """
        Save model, scaler, and feature names to disk
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Save Keras model
        self.model.save(self.model_path)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Save feature names
        with open(self.feature_names_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"✅ Model saved to {self.model_path}")
        print(f"✅ Scaler saved to {self.scaler_path}")
        print(f"✅ Feature names saved to {self.feature_names_path}")
    
    def load_model(self):
        """
        Load model, scaler, and feature names from disk
        
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✅ Neural Network model loaded from {self.model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                return False
        else:
            print(f"⚠️  No saved model found at {self.model_path}")
            return False
        
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            print(f"✅ Scaler loaded from {self.scaler_path}")
        
        if os.path.exists(self.feature_names_path):
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"✅ Feature names loaded: {len(self.feature_names)} features")
        
        return True
    
    def is_trained(self):
        """
        Check if model is trained and ready
        
        Returns:
            Boolean
        """
        return self.model is not None and self.scaler is not None
    
    def get_model_info(self):
        """
        Get information about the trained model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"trained": False, "message": "Model not trained yet"}
        
        return {
            "trained": True,
            "model_type": "Neural Network (Keras/TensorFlow)",
            "architecture": "Sequential (4 hidden layers)",
            "layers": len(self.model.layers),
            "total_params": self.model.count_params(),
            "n_features": len(self.feature_names),
            "features": self.feature_names,
            "model_path": self.model_path
        }


# ===== Helper Functions =====

def get_clinical_recommendations(patient_data, risk_score):
    """
    Generate clinical recommendations based on patient data and risk
    
    Args:
        patient_data: Dictionary with patient information
        risk_score: Float risk score (0-1)
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # High risk alerts
    if risk_score >= 0.7:
        recommendations.append("⚠️ URGENT: Immediate clinical review required")
        recommendations.append("Consider ICU transfer or escalation of care")
    
    # Vital signs checks
    if patient_data.get('spo2_pct', 100) < 92:
        recommendations.append("Monitor oxygen saturation closely - Consider increasing O2 support")
    
    if patient_data.get('heart_rate', 80) > 120 or patient_data.get('heart_rate', 80) < 50:
        recommendations.append("Abnormal heart rate detected - Cardiac monitoring recommended")
    
    if patient_data.get('respiratory_rate', 16) > 24:
        recommendations.append("Elevated respiratory rate - Assess for respiratory distress")
    
    systolic = patient_data.get('systolic_bp', 120)
    if systolic < 90 or systolic > 180:
        recommendations.append("Blood pressure out of normal range - Immediate assessment needed")
    
    # Lab values checks
    if patient_data.get('lactate', 1) > 2.0:
        recommendations.append("Elevated lactate - Consider sepsis workup")
    
    if patient_data.get('creatinine', 1) > 1.5:
        recommendations.append("Elevated creatinine - Monitor renal function")
    
    temp = patient_data.get('temperature_c', 37)
    if temp > 38.5 or temp < 36:
        recommendations.append("Temperature abnormality - Investigate cause")
    
    if patient_data.get('wbc_count', 8) > 12 or patient_data.get('wbc_count', 8) < 4:
        recommendations.append("Abnormal WBC count - Monitor for infection")
    
    # Low risk
    if risk_score < 0.4 and len(recommendations) == 0:
        recommendations.append("Continue routine monitoring")
        recommendations.append("Patient appears stable")
    
    return recommendations


# ===== Example Usage =====

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 Neural Network Patient Risk Prediction - Test")
    print("="*60 + "\n")
    
    # Initialize model
    model = PatientRiskModel()
    
    # Check if model is already trained
    if not model.is_trained():
        print("⚠️  Model not trained. Please train first using:")
        print("   model.train(data_path='path/to/data.csv')")
    else:
        print("✅ Neural Network model is ready!")
        
        # Get model info
        info = model.get_model_info()
        print(f"\nModel Type: {info['model_type']}")
        print(f"Architecture: {info['architecture']}")
        print(f"Layers: {info['layers']}")
        print(f"Total Parameters: {info['total_params']:,}")
        print(f"Features: {info['n_features']}")
        
        # Example prediction - with text values to test encoding
        sample_patient = {
            'age': 65,
            'gender': 'M',              # نص سيتحول لـ 1
            'heart_rate': 95,
            'respiratory_rate': 18,
            'spo2_pct': 96,
            'temperature_c': 37.2,
            'systolic_bp': 130,
            'diastolic_bp': 80,
            'wbc_count': 8.5,
            'lactate': 1.2,
            'creatinine': 1.1,
            'crp_level': 5,
            'hemoglobin': 13.5,
            'oxygen_flow': 2,
            'oxygen_device': 'nasal',   # نص سيتحول لـ 1
            'nurse_alert': 0,
            'mobility_score': 3,
            'comorbidity_index': 2
        }
        
        print("\n" + "="*60)
        print("🔍 Sample Prediction")
        print("="*60)
        
        result = model.predict(sample_patient)
        print(f"\nRisk Score: {result['risk_score']*100:.1f}%")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Prediction: {'Deterioration' if result['prediction'] == 1 else 'Stable'}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        
        # Get recommendations
        recommendations = get_clinical_recommendations(sample_patient, result['risk_score'])
        print("\n📋 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")