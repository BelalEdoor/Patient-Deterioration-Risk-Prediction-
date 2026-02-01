import React, { useState, useEffect } from 'react';
import './PredictPatient.css';

function PredictPatient({ modelStatus: modelStatusProp }) {
  const [formData, setFormData] = useState({
    patient_id: '',
    age: '',
    gender: '1',
    heart_rate: '',
    respiratory_rate: '',
    spo2_pct: '',
    temperature_c: '',
    systolic_bp: '',
    diastolic_bp: '',
    wbc_count: '',
    lactate: '',
    creatinine: '',
    crp_level: '',
    hemoglobin: '',
    oxygen_flow: '',
    oxygen_device: '0',
    nurse_alert: '0',
    mobility_score: '2',
    comorbidity_index: '0'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [validationErrors, setValidationErrors] = useState({});

  const [modelStatus, setModelStatus] = useState(modelStatusProp || null);

  // ===== الحقول الإلزامية التي يجب أن يملأها المستخدم =====
  const requiredFields = {
    patient_id: 'معرف المريض (Patient ID)',
    age: 'العمر (Age)',
    heart_rate: 'معدل القلب (Heart Rate)',
    respiratory_rate: 'معدل التنفس (Respiratory Rate)',
    spo2_pct: 'نسبة الأكسجين (SpO2)',
    temperature_c: 'درجة الحرارة (Temperature)',
    systolic_bp: 'ضغط الدم الانقباضي (Systolic BP)',
    diastolic_bp: 'ضغط الدم الانبسطي (Diastolic BP)',
    wbc_count: 'عدد خلايا الدم البيضاء (WBC)',
    lactate: 'اللاكتات (Lactate)',
    creatinine: 'الكريتينين (Creatinine)',
    crp_level: 'بروتين سي الفعّال (CRP)',
    hemoglobin: 'الهيموجلوبين (Hemoglobin)',
    oxygen_flow: 'معدل تدفق الأكسجين (O2 Flow)'
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
    // إزالة خطأ الحقل عند الكتابة فيه
    if (validationErrors[name]) {
      setValidationErrors(prev => {
        const updated = { ...prev };
        delete updated[name];
        return updated;
      });
    }
  };

  useEffect(() => {
    const fetchModelStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/model/status");
        const data = await res.json();
        setModelStatus(data);
      } catch (err) {
        console.error("Failed to fetch model status", err);
      }
    };

    fetchModelStatus();

    const interval = setInterval(fetchModelStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  // ===== دالة التحقق من البيانات قبل الإرسال =====
  const validateForm = () => {
    const errors = {};

    // تحقق من الحقول الفارغة
    Object.keys(requiredFields).forEach((field) => {
      if (!formData[field] || formData[field].toString().trim() === '') {
        errors[field] = `الحقل إلزامي - ${requiredFields[field]}`;
      }
    });

    // تحقق من القيم الرقمية (كل شي غير patient_id يجب يكون رقم)
    const numericFields = Object.keys(requiredFields).filter(f => f !== 'patient_id');
    numericFields.forEach((field) => {
      if (formData[field] && isNaN(parseFloat(formData[field]))) {
        errors[field] = `يجب أن يكون رقم - Must be a number`;
      }
    });

    // تحقق من النطاقات المسموحة (من Pydantic في البـ Backend)
    const ranges = {
      age:                { min: 0,    max: 120,  label: '0 - 120' },
      heart_rate:         { min: 30,   max: 200,  label: '30 - 200 bpm' },
      respiratory_rate:   { min: 5,    max: 60,   label: '5 - 60' },
      spo2_pct:           { min: 70,   max: 100,  label: '70 - 100%' },
      temperature_c:      { min: 35,   max: 42,   label: '35 - 42°C' },
      systolic_bp:        { min: 60,   max: 250,  label: '60 - 250 mmHg' },
      diastolic_bp:       { min: 30,   max: 150,  label: '30 - 150 mmHg' },
      wbc_count:          { min: 1,    max: 50,   label: '1 - 50' },
      lactate:            { min: 0,    max: 20,   label: '0 - 20 mmol/L' },
      creatinine:         { min: 0.1,  max: 15,   label: '0.1 - 15 mg/dL' },
      crp_level:          { min: 0,    max: 500,  label: '0 - 500 mg/L' },
      hemoglobin:         { min: 4,    max: 20,   label: '4 - 20 g/dL' },
      oxygen_flow:        { min: 0,    max: 15,   label: '0 - 15 L/min' }
    };

    Object.keys(ranges).forEach((field) => {
      const val = parseFloat(formData[field]);
      if (!isNaN(val)) {
        const { min, max, label } = ranges[field];
        if (val < min || val > max) {
          errors[field] = `النطاق المسموح: ${label}`;
        }
      }
    });

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!modelStatus?.trained) {
      setError('النموذج غير جاهز حالياً. يرجى المحاولة لاحقاً - Model is not ready. Please try again later.');
      return;
    }

    // ===== التحقق من البيانات قبل الإرسال =====
    if (!validateForm()) {
      setError('يوجد أخطاء في البيانات. يرجى مراجعة الحقول الحمراء - Please fix the highlighted fields.');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const payload = {
        patient_id: formData.patient_id.trim(),
        age: parseFloat(formData.age),
        gender: parseInt(formData.gender),
        heart_rate: parseFloat(formData.heart_rate),
        respiratory_rate: parseFloat(formData.respiratory_rate),
        spo2_pct: parseFloat(formData.spo2_pct),
        temperature_c: parseFloat(formData.temperature_c),
        systolic_bp: parseFloat(formData.systolic_bp),
        diastolic_bp: parseFloat(formData.diastolic_bp),
        wbc_count: parseFloat(formData.wbc_count),
        lactate: parseFloat(formData.lactate),
        creatinine: parseFloat(formData.creatinine),
        crp_level: parseFloat(formData.crp_level),
        hemoglobin: parseFloat(formData.hemoglobin),
        oxygen_flow: parseFloat(formData.oxygen_flow),
        oxygen_device: parseInt(formData.oxygen_device),
        nurse_alert: parseInt(formData.nurse_alert),
        mobility_score: parseInt(formData.mobility_score),
        comorbidity_index: parseInt(formData.comorbidity_index)
      };

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        // عرض تفاصيل الخطأ من البـ Backend
        const detail = errorData.detail;
        if (Array.isArray(detail)) {
          // Pydantic يرجع validation errors كـ array
          const messages = detail.map(d => `${d.loc?.join(' → ')} : ${d.msg}`).join('\n');
          throw new Error(messages);
        }
        throw new Error(detail || 'Prediction failed');
      }

      const data = await response.json();
      setPrediction(data);

      const savedPredictions = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
      savedPredictions.unshift(data);
      localStorage.setItem('predictionHistory', JSON.stringify(savedPredictions.slice(0, 100)));

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (category) => {
    if (category.includes('High')) return '#ef4444';
    if (category.includes('Medium')) return '#f59e0b';
    return '#10b981';
  };

  const fillSampleData = () => {
    setFormData({
      patient_id: 'P' + Math.floor(Math.random() * 10000).toString().padStart(4, '0'),
      age: '65',
      gender: '1',
      heart_rate: '95',
      respiratory_rate: '18',
      spo2_pct: '96',
      temperature_c: '37.2',
      systolic_bp: '130',
      diastolic_bp: '80',
      wbc_count: '8.5',
      lactate: '1.2',
      creatinine: '1.1',
      crp_level: '5',
      hemoglobin: '13.5',
      oxygen_flow: '2',
      oxygen_device: '1',
      nurse_alert: '0',
      mobility_score: '3',
      comorbidity_index: '2'
    });
    setValidationErrors({});
    setError(null);
  };

  const resetForm = () => {
    setFormData({
      patient_id: '',
      age: '',
      gender: '1',
      heart_rate: '',
      respiratory_rate: '',
      spo2_pct: '',
      temperature_c: '',
      systolic_bp: '',
      diastolic_bp: '',
      wbc_count: '',
      lactate: '',
      creatinine: '',
      crp_level: '',
      hemoglobin: '',
      oxygen_flow: '',
      oxygen_device: '0',
      nurse_alert: '0',
      mobility_score: '2',
      comorbidity_index: '0'
    });
    setPrediction(null);
    setError(null);
    setValidationErrors({});
  };

  return (
    <div className="predict-patient">
      <div className="page-header">
        <h2>🔮 التنبؤ بخطر تدهور المريض</h2>
        <p>Predict Patient Deterioration Risk</p>
      </div>

      {modelStatus?.trained ? (
        <div className="success-banner">
          ✅ النموذج جاهز للتنبؤ | Model is ready
        </div>
      ) : (
        <div className="warning-banner">
          ⏳ جاري تدريب النموذج... | Training model, please wait
        </div>
      )}

      <div className="predict-container">
        <div className="form-section">
          <div className="section-header">
            <h3>📋 معلومات المريض - Patient Information</h3>
            <div className="header-buttons">
              <button type="button" className="btn-secondary" onClick={fillSampleData}>
                📝 بيانات تجريبية - Sample Data
              </button>
              <button type="button" className="btn-secondary" onClick={resetForm}>
                🔄 مسح - Clear
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit}>

            {/* === معرف المريض === */}
            <div className="form-group">
              <label>معرف المريض - Patient ID *</label>
              <input
                type="text"
                name="patient_id"
                value={formData.patient_id}
                onChange={handleChange}
                placeholder="مثال: P0001"
                className={validationErrors.patient_id ? 'input-error' : ''}
              />
              {validationErrors.patient_id && <span className="error-text">{validationErrors.patient_id}</span>}
            </div>

            {/* === المعلومات الأساسية === */}
            <div className="form-section-title">👤 المعلومات الأساسية - Basic Info</div>
            <div className="form-row">
              <div className="form-group">
                <label>العمر - Age *</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="0 - 120"
                  className={validationErrors.age ? 'input-error' : ''}
                />
                {validationErrors.age && <span className="error-text">{validationErrors.age}</span>}
              </div>
              <div className="form-group">
                <label>الجنس - Gender</label>
                <select name="gender" value={formData.gender} onChange={handleChange}>
                  <option value="1">ذكر - Male</option>
                  <option value="0">أنثى - Female</option>
                </select>
              </div>
            </div>

            {/* === علامات حيوية === */}
            <div className="form-section-title">❤️ العلامات الحيوية - Vital Signs</div>
            <div className="form-row">
              <div className="form-group">
                <label>معدل القلب - Heart Rate (bpm) *</label>
                <input
                  type="number"
                  name="heart_rate"
                  value={formData.heart_rate}
                  onChange={handleChange}
                  placeholder="30 - 200"
                  className={validationErrors.heart_rate ? 'input-error' : ''}
                />
                {validationErrors.heart_rate && <span className="error-text">{validationErrors.heart_rate}</span>}
              </div>
              <div className="form-group">
                <label>معدل التنفس - Resp. Rate *</label>
                <input
                  type="number"
                  name="respiratory_rate"
                  value={formData.respiratory_rate}
                  onChange={handleChange}
                  placeholder="5 - 60"
                  className={validationErrors.respiratory_rate ? 'input-error' : ''}
                />
                {validationErrors.respiratory_rate && <span className="error-text">{validationErrors.respiratory_rate}</span>}
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>نسبة الأكسجين - SpO2 (%) *</label>
                <input
                  type="number"
                  name="spo2_pct"
                  value={formData.spo2_pct}
                  onChange={handleChange}
                  placeholder="70 - 100"
                  className={validationErrors.spo2_pct ? 'input-error' : ''}
                />
                {validationErrors.spo2_pct && <span className="error-text">{validationErrors.spo2_pct}</span>}
              </div>
              <div className="form-group">
                <label>درجة الحرارة - Temp (°C) *</label>
                <input
                  type="number"
                  name="temperature_c"
                  value={formData.temperature_c}
                  onChange={handleChange}
                  placeholder="35 - 42"
                  step="0.1"
                  className={validationErrors.temperature_c ? 'input-error' : ''}
                />
                {validationErrors.temperature_c && <span className="error-text">{validationErrors.temperature_c}</span>}
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>ضغط الدم الانقباضي - Systolic BP *</label>
                <input
                  type="number"
                  name="systolic_bp"
                  value={formData.systolic_bp}
                  onChange={handleChange}
                  placeholder="60 - 250"
                  className={validationErrors.systolic_bp ? 'input-error' : ''}
                />
                {validationErrors.systolic_bp && <span className="error-text">{validationErrors.systolic_bp}</span>}
              </div>
              <div className="form-group">
                <label>ضغط الدم الانبسطي - Diastolic BP *</label>
                <input
                  type="number"
                  name="diastolic_bp"
                  value={formData.diastolic_bp}
                  onChange={handleChange}
                  placeholder="30 - 150"
                  className={validationErrors.diastolic_bp ? 'input-error' : ''}
                />
                {validationErrors.diastolic_bp && <span className="error-text">{validationErrors.diastolic_bp}</span>}
              </div>
            </div>

            {/* === نتائج المخبر === */}
            <div className="form-section-title">🔬 نتائج المخبر - Lab Results</div>
            <div className="form-row">
              <div className="form-group">
                <label>خلايا الدم البيضاء - WBC *</label>
                <input
                  type="number"
                  name="wbc_count"
                  value={formData.wbc_count}
                  onChange={handleChange}
                  placeholder="1 - 50"
                  step="0.1"
                  className={validationErrors.wbc_count ? 'input-error' : ''}
                />
                {validationErrors.wbc_count && <span className="error-text">{validationErrors.wbc_count}</span>}
              </div>
              <div className="form-group">
                <label>اللاكتات - Lactate (mmol/L) *</label>
                <input
                  type="number"
                  name="lactate"
                  value={formData.lactate}
                  onChange={handleChange}
                  placeholder="0 - 20"
                  step="0.1"
                  className={validationErrors.lactate ? 'input-error' : ''}
                />
                {validationErrors.lactate && <span className="error-text">{validationErrors.lactate}</span>}
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>الكريتينين - Creatinine (mg/dL) *</label>
                <input
                  type="number"
                  name="creatinine"
                  value={formData.creatinine}
                  onChange={handleChange}
                  placeholder="0.1 - 15"
                  step="0.1"
                  className={validationErrors.creatinine ? 'input-error' : ''}
                />
                {validationErrors.creatinine && <span className="error-text">{validationErrors.creatinine}</span>}
              </div>
              <div className="form-group">
                <label>بروتين سي الفعّال - CRP (mg/L) *</label>
                <input
                  type="number"
                  name="crp_level"
                  value={formData.crp_level}
                  onChange={handleChange}
                  placeholder="0 - 500"
                  step="0.1"
                  className={validationErrors.crp_level ? 'input-error' : ''}
                />
                {validationErrors.crp_level && <span className="error-text">{validationErrors.crp_level}</span>}
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>الهيموجلوبين - Hemoglobin (g/dL) *</label>
                <input
                  type="number"
                  name="hemoglobin"
                  value={formData.hemoglobin}
                  onChange={handleChange}
                  placeholder="4 - 20"
                  step="0.1"
                  className={validationErrors.hemoglobin ? 'input-error' : ''}
                />
                {validationErrors.hemoglobin && <span className="error-text">{validationErrors.hemoglobin}</span>}
              </div>
            </div>

            {/* === التدخلات السريرية === */}
            <div className="form-section-title">🏥 التدخلات السريرية - Clinical Interventions</div>
            <div className="form-row">
              <div className="form-group">
                <label>تدفق الأكسجين - O2 Flow (L/min) *</label>
                <input
                  type="number"
                  name="oxygen_flow"
                  value={formData.oxygen_flow}
                  onChange={handleChange}
                  placeholder="0 - 15"
                  step="0.1"
                  className={validationErrors.oxygen_flow ? 'input-error' : ''}
                />
                {validationErrors.oxygen_flow && <span className="error-text">{validationErrors.oxygen_flow}</span>}
              </div>
              <div className="form-group">
                <label>جهاز الأكسجين - O2 Device</label>
                <select name="oxygen_device" value={formData.oxygen_device} onChange={handleChange}>
                  <option value="0">بدون جهاز - None</option>
                  <option value="1">أنبوب أنفي - Nasal Cannula</option>
                  <option value="2">قناع وجه - Face Mask</option>
                  <option value="3">أجهزة التنفس - Ventilator</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>تنبيه الممرض - Nurse Alert</label>
                <select name="nurse_alert" value={formData.nurse_alert} onChange={handleChange}>
                  <option value="0">لا - No</option>
                  <option value="1">نعم - Yes</option>
                </select>
              </div>
              <div className="form-group">
                <label>درجة الحركة - Mobility Score</label>
                <select name="mobility_score" value={formData.mobility_score} onChange={handleChange}>
                  <option value="0">طريح - Bedridden</option>
                  <option value="1">حركة محدودة - Limited</option>
                  <option value="2">حركة جزئية - Partial</option>
                  <option value="3">حركة شبه كاملة - Mostly Mobile</option>
                  <option value="4">حركة كاملة - Fully Mobile</option>
                </select>
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>عدد الأمراض المزمنة - Comorbidities</label>
                <select name="comorbidity_index" value={formData.comorbidity_index} onChange={handleChange}>
                  {[...Array(11).keys()].map(i => (
                    <option key={i} value={String(i)}>{i}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-actions">
              <button 
                type="submit" 
                className="btn-primary btn-large"
                disabled={loading || !modelStatus?.trained}
              >
                {loading ? '🔄 جاري التحليل... Analyzing...' : '🎯 تنبؤ بالخطر - Predict Risk'}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        <div className="results-section">
          {error && (
            <div className="error-card">
              <div className="error-icon">❌</div>
              <h3>خطأ - Error</h3>
              <p style={{ whiteSpace: 'pre-line' }}>{error}</p>
            </div>
          )}

          {prediction && (
            <div className="prediction-result fade-in">
              <div className="result-header">
                <h3>📊 نتائج التنبؤ - Results</h3>
                <span className="patient-badge">{prediction.patient_id}</span>
              </div>

              <div 
                className="risk-indicator"
                style={{ background: `linear-gradient(135deg, ${getRiskColor(prediction.risk_category)}, ${getRiskColor(prediction.risk_category)}dd)` }}
              >
                <div className="risk-icon">
                  {prediction.risk_category.includes('High') && '🚨'}
                  {prediction.risk_category.includes('Medium') && '⚠️'}
                  {prediction.risk_category.includes('Low') && '✅'}
                </div>
                <h2>{prediction.risk_category}</h2>
                <div className="risk-score-display">
                  {(prediction.risk_score * 100).toFixed(1)}%
                </div>
                <p className="risk-label">درجة الخطر - Risk Score</p>
              </div>

              <div className="prediction-details">
                <div className="detail-row">
                  <span className="label">الثقة - Confidence:</span>
                  <span className="value">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="detail-row">
                  <span className="label">التنبؤ - Prediction:</span>
                  <span className="value">
                    {prediction.prediction === 1 
                      ? '⚠️ تدهور متوقع - Deterioration Expected' 
                      : '✅ مستقر - Stable'}
                  </span>
                </div>
                <div className="detail-row">
                  <span className="label">التاريخ - Timestamp:</span>
                  <span className="value">{new Date(prediction.timestamp).toLocaleString('ar-PS')}</span>
                </div>
              </div>

              <div className="recommendations-card">
                <h4>📋 التوصيات السريرية - Clinical Recommendations</h4>
                <ul className="recommendations-list">
                  {prediction.recommendations.map((rec, index) => (
                    <li key={index}>
                      <span className="rec-bullet">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="action-buttons">
                <button className="btn-secondary" onClick={resetForm}>
                  🔄 تنبؤ جديد - New Prediction
                </button>
                <a href="/history" className="btn-view">
                  📚 عرض السجل - View History
                </a>
              </div>
            </div>
          )}

          {!prediction && !error && (
            <div className="placeholder-card">
              <div className="placeholder-icon">🔮</div>
              <h3>جاهز للتنبؤ - Ready to Predict</h3>
              <p>املأ بيانات المريض واضغط على "تنبؤ بالخطر" لرؤية النتائج</p>
              <p>Fill in patient data and click "Predict Risk" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default PredictPatient;