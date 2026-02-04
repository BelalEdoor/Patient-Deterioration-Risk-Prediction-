import React, { useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './PredictPatient.css';

function CSVUpload({ modelStatus }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const printRef = useRef();
  const navigate = useNavigate();

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        setError('الرجاء اختيار ملف CSV فقط / Please select a CSV file only');
        setSelectedFile(null);
        return;
      }
      setSelectedFile(file);
      setError(null);
      setResult(null);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.csv')) {
        setSelectedFile(file);
        setError(null);
        setResult(null);
      } else {
        setError('الرجاء اختيار ملف CSV فقط / Please select a CSV file only');
      }
    }
  };

  // Advanced analysis function
  const analyzePatientData = (csvData) => {
    const lines = csvData.split('\n').filter(line => line.trim());
    const headers = lines[0].split(',');
    
    const readings = [];
    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      if (values.length === headers.length) {
        const reading = {};
        headers.forEach((header, index) => {
          const value = values[index].trim();
          reading[header] = isNaN(value) ? value : parseFloat(value);
        });
        readings.push(reading);
      }
    }

    if (readings.length === 0) {
      throw new Error('لا توجد بيانات صالحة في الملف / No valid data in file');
    }

    // Calculate risk scores with advanced algorithm
    const hourlyReadings = readings.map(reading => {
      let riskScore = 0;
      let riskFactors = [];

      // Vital Signs Assessment
      // SpO2 (Oxygen Saturation)
      if (reading.spo2_pct < 90) {
        riskScore += 0.25;
        riskFactors.push('Critical SpO2');
      } else if (reading.spo2_pct < 92) {
        riskScore += 0.15;
        riskFactors.push('Low SpO2');
      } else if (reading.spo2_pct < 95) {
        riskScore += 0.08;
        riskFactors.push('Borderline SpO2');
      }

      // Heart Rate
      if (reading.heart_rate > 120 || reading.heart_rate < 50) {
        riskScore += 0.20;
        riskFactors.push('Abnormal HR');
      } else if (reading.heart_rate > 100 || reading.heart_rate < 60) {
        riskScore += 0.10;
        riskFactors.push('Elevated HR');
      }

      // Respiratory Rate
      if (reading.respiratory_rate > 25 || reading.respiratory_rate < 10) {
        riskScore += 0.20;
        riskFactors.push('Abnormal RR');
      } else if (reading.respiratory_rate > 20) {
        riskScore += 0.10;
        riskFactors.push('Elevated RR');
      }

      // Temperature
      if (reading.temperature_c > 38.5 || reading.temperature_c < 35.5) {
        riskScore += 0.15;
        riskFactors.push('Abnormal Temp');
      } else if (reading.temperature_c > 37.8) {
        riskScore += 0.08;
        riskFactors.push('Fever');
      }

      // Blood Pressure
      if (reading.systolic_bp < 90 || reading.systolic_bp > 160) {
        riskScore += 0.18;
        riskFactors.push('Abnormal BP');
      }

      // Laboratory Values
      // Lactate
      if (reading.lactate > 4.0) {
        riskScore += 0.25;
        riskFactors.push('High Lactate');
      } else if (reading.lactate > 2.5) {
        riskScore += 0.15;
        riskFactors.push('Elevated Lactate');
      }

      // WBC Count
      if (reading.wbc_count > 15 || reading.wbc_count < 4) {
        riskScore += 0.15;
        riskFactors.push('Abnormal WBC');
      }

      // Creatinine
      if (reading.creatinine > 2.0) {
        riskScore += 0.20;
        riskFactors.push('Renal Impairment');
      } else if (reading.creatinine > 1.5) {
        riskScore += 0.10;
        riskFactors.push('Elevated Creatinine');
      }

      // CRP (C-Reactive Protein)
      if (reading.crp_level > 100) {
        riskScore += 0.20;
        riskFactors.push('High Inflammation');
      } else if (reading.crp_level > 50) {
        riskScore += 0.12;
        riskFactors.push('Inflammation');
      }

      // Hemoglobin
      if (reading.hemoglobin < 8 || reading.hemoglobin > 18) {
        riskScore += 0.15;
        riskFactors.push('Abnormal Hb');
      }

      // Oxygen Support
      if (reading.oxygen_device >= 2) {
        riskScore += 0.15;
        riskFactors.push('High O2 Support');
      } else if (reading.oxygen_flow > 0) {
        riskScore += 0.08;
        riskFactors.push('O2 Support');
      }

      // Mobility Score
      if (reading.mobility_score <= 1) {
        riskScore += 0.15;
        riskFactors.push('Poor Mobility');
      }

      // Comorbidity Index
      if (reading.comorbidity_index >= 4) {
        riskScore += 0.12;
        riskFactors.push('Multiple Comorbidities');
      }

      // Nurse Alert
      if (reading.nurse_alert === 1) {
        riskScore += 0.10;
        riskFactors.push('Nurse Concern');
      }

      // Cap risk score at 1.0
      riskScore = Math.min(riskScore, 1.0);

      return {
        hour: reading.hour,
        risk_score: riskScore,
        risk_factors: riskFactors,
        spo2: reading.spo2_pct,
        heart_rate: reading.heart_rate,
        respiratory_rate: reading.respiratory_rate,
        temperature: reading.temperature_c,
        lactate: reading.lactate,
        systolic_bp: reading.systolic_bp,
        diastolic_bp: reading.diastolic_bp,
        wbc: reading.wbc_count,
        creatinine: reading.creatinine,
        crp: reading.crp_level,
        hemoglobin: reading.hemoglobin,
        oxygen_flow: reading.oxygen_flow,
        oxygen_device: reading.oxygen_device
      };
    });

    // Calculate statistics
    const riskScores = hourlyReadings.map(r => r.risk_score);
    const minRisk = Math.min(...riskScores);
    const maxRisk = Math.max(...riskScores);
    const avgRisk = riskScores.reduce((a, b) => a + b, 0) / riskScores.length;
    
    // Determine trend
    const firstHalf = riskScores.slice(0, Math.floor(riskScores.length / 2));
    const secondHalf = riskScores.slice(Math.floor(riskScores.length / 2));
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    
    let trend = 'stable';
    if (secondAvg > firstAvg + 0.1) trend = 'increasing';
    else if (secondAvg < firstAvg - 0.1) trend = 'decreasing';

    // Detect deterioration
    let deteriorationDetected = false;
    let deteriorationHour = null;
    for (let i = 1; i < hourlyReadings.length; i++) {
      const increase = hourlyReadings[i].risk_score - hourlyReadings[i - 1].risk_score;
      if (increase > 0.15) {
        deteriorationDetected = true;
        deteriorationHour = hourlyReadings[i].hour;
        break;
      }
    }

    // Generate recommendations
    const recommendations = generateRecommendations(hourlyReadings, maxRisk, trend);

    // Get patient info
    const patientId = readings[0].patient_id || 'Unknown';
    const age = readings[0].age;
    const gender = readings[0].gender;

    return {
      patient_id: patientId,
      patient_age: age,
      patient_gender: gender === 1 ? 'Male' : gender === 0 ? 'Female' : 'Unknown',
      total_hours: hourlyReadings.length,
      final_risk_score: riskScores[riskScores.length - 1],
      hourly_readings: hourlyReadings,
      summary: {
        min_risk: minRisk,
        max_risk: maxRisk,
        avg_risk: avgRisk,
        risk_trend: trend,
        hours_high_risk: riskScores.filter(r => r >= 0.7).length,
        hours_medium_risk: riskScores.filter(r => r >= 0.4 && r < 0.7).length,
        hours_low_risk: riskScores.filter(r => r < 0.4).length
      },
      deterioration_detected: deteriorationDetected,
      deterioration_hour: deteriorationHour,
      recommendations: recommendations,
      timestamp: new Date().toISOString()
    };
  };

  const generateRecommendations = (readings, maxRisk, trend) => {
    const recommendations = [];
    const lastReading = readings[readings.length - 1];

    if (maxRisk >= 0.7) {
      recommendations.push('🚨 Critical: Immediate medical intervention required');
      recommendations.push('Monitor oxygen saturation closely - Consider increasing O2 support');
    }

    if (lastReading.spo2 < 92) {
      recommendations.push('Abnormal oxygen saturation detected - Assess respiratory distress');
    }

    if (lastReading.lactate > 4.0) {
      recommendations.push('Elevated lactate levels - Investigate cause and monitor closely');
    }

    if (lastReading.heart_rate > 120) {
      recommendations.push('Tachycardia detected - Cardiac monitoring recommended');
    }

    if (lastReading.temperature >= 38.5) {
      recommendations.push('Fever present - Blood cultures and assess for sepsis');
    }

    if (lastReading.respiratory_rate > 25) {
      recommendations.push('Abnormal respiratory rate - Assess respiratory distress');
    }

    if (lastReading.creatinine > 2.0) {
      recommendations.push('Elevated creatinine - Monitor renal function closely');
    }

    if (lastReading.crp > 100) {
      recommendations.push('Significant inflammation detected - Immediate assessment needed');
    }

    if (trend === 'increasing') {
      recommendations.push('⚠️ Temperature abnormality - Investigate cause');
      recommendations.push('Elevated creatinine - Monitor renal function');
    }

    if (lastReading.systolic_bp < 90) {
      recommendations.push('Blood pressure out of normal range - Immediate assessment needed');
    }

    if (recommendations.length === 0) {
      recommendations.push('✅ Patients health condition is stable with low risk level');
      recommendations.push('Continue routine monitoring');
    }

    return recommendations;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('الرجاء اختيار ملف أولاً / Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const fileContent = await selectedFile.text();
      const analysisResult = analyzePatientData(fileContent);
      setResult(analysisResult);
      
      // Save to localStorage for dashboard
      saveToDashboard(analysisResult);
      
      console.log('Analysis result:', analysisResult);
    } catch (err) {
      setError(err.message || 'حدث خطأ في معالجة الملف / Error processing file');
      console.error('Analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const saveToDashboard = (analysisResult) => {
    try {
      // Save each hourly reading as a separate prediction
      const existingPredictions = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
      
      // Create a summary prediction for the final state
      const summaryPrediction = {
        patient_id: analysisResult.patient_id,
        risk_score: analysisResult.final_risk_score,
        prediction: analysisResult.final_risk_score >= 0.5 ? 1 : 0,
        confidence: 0.85,
        timestamp: analysisResult.timestamp,
        input_data: {
          age: analysisResult.patient_age,
          gender: analysisResult.patient_gender === 'Male' ? 1 : 0,
          heart_rate: analysisResult.hourly_readings[analysisResult.hourly_readings.length - 1].heart_rate,
          spo2_pct: analysisResult.hourly_readings[analysisResult.hourly_readings.length - 1].spo2,
          temperature: analysisResult.hourly_readings[analysisResult.hourly_readings.length - 1].temperature,
          lactate: analysisResult.hourly_readings[analysisResult.hourly_readings.length - 1].lactate
        },
        recommendations: analysisResult.recommendations,
        analysis_summary: analysisResult.summary
      };

      existingPredictions.unshift(summaryPrediction);
      localStorage.setItem('predictionHistory', JSON.stringify(existingPredictions));
    } catch (err) {
      console.error('Error saving to dashboard:', err);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const getRiskColor = (score) => {
    if (score >= 0.7) return '#ef4444';
    if (score >= 0.5) return '#f59e0b';
    if (score >= 0.3) return '#eab308';
    return '#10b981';
  };

  const getRiskGradient = (score) => {
    if (score >= 0.7) return 'linear-gradient(135deg, #ef4444, #dc2626)';
    if (score >= 0.5) return 'linear-gradient(135deg, #f59e0b, #d97706)';
    if (score >= 0.3) return 'linear-gradient(135deg, #eab308, #ca8a04)';
    return 'linear-gradient(135deg, #10b981, #059669)';
  };

  const getRiskLabel = (score) => {
    if (score >= 0.7) return { ar: 'خطر عالي', en: 'High Risk', icon: '🔴' };
    if (score >= 0.5) return { ar: 'خطر متوسط', en: 'Medium Risk', icon: '🟠' };
    if (score >= 0.3) return { ar: 'خطر منخفض-متوسط', en: 'Low-Medium Risk', icon: '🟡' };
    return { ar: 'خطر منخفض', en: 'Low Risk', icon: '🟢' };
  };

  const getTrendIcon = (trend) => {
    if (trend === 'increasing') return '📈';
    if (trend === 'decreasing') return '📉';
    return '➡️';
  };

  const getTrendLabel = (trend) => {
    if (trend === 'increasing') return { ar: 'متزايد', en: 'Increasing' };
    if (trend === 'decreasing') return { ar: 'متناقص', en: 'Decreasing' };
    return { ar: 'مستقر', en: 'Stable' };
  };

  const hasSignificantDeterioration = () => {
    if (!result || !result.deterioration_detected) return false;
    if (result.final_risk_score >= 0.5) return true;
    const maxRisk = result.summary?.max_risk || 0;
    const avgRisk = result.summary?.avg_risk || 0;
    return (maxRisk - avgRisk) > 0.3;
  };

  return (
    <div className="predict-patient">
      <div className="page-header no-print">
        <div className="header-icon">📊</div>
        <h2>تحليل ملف مريض - CSV File Analysis</h2>
        <p className="header-subtitle">
          قم برفع ملف CSV يحتوي على قراءات المريض لعدة ساعات للحصول على تحليل شامل للحالة الصحية
        </p>
        <p className="header-subtitle-en">
          Upload a CSV file containing patient readings across multiple hours for comprehensive health analysis
        </p>
      </div>

      {!modelStatus?.trained && (
        <div className="warning-banner no-print">
          <span className="warning-icon">⚠️</span>
          <div>
            <strong>النموذج غير مدرب!</strong> الرجاء التأكد من تدريب النموذج أولاً
            <br />
            <strong>Model not trained!</strong> Please ensure the model is trained first
          </div>
        </div>
      )}

      <div className="predict-container">
        {/* Upload Section */}
        <div className="form-section no-print">
          <div className="section-header">
            <h3>📁 رفع ملف CSV</h3>
          </div>

          <div 
            className={`upload-area ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="upload-box">
              <div className="upload-icon">📄</div>
              <h4>اسحب وأفلت ملف CSV هنا</h4>
              <p className="upload-subtitle">أو انقر للاختيار من جهازك</p>
              <p className="upload-subtitle-en">Drag & drop CSV file here or click to select</p>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileSelect}
                className="file-input"
                id="csv-upload"
                disabled={loading}
              />
              <label htmlFor="csv-upload" className="file-label btn btn-primary">
                <span className="btn-icon">📂</span>
                اختر ملف CSV
              </label>
            </div>

            {selectedFile && (
              <div className="selected-file">
                <div className="file-info">
                  <div className="file-icon">📄</div>
                  <div>
                    <strong>{selectedFile.name}</strong>
                    <p>{(selectedFile.size / 1024).toFixed(2)} KB</p>
                  </div>
                </div>
                <button className="btn-remove" onClick={() => setSelectedFile(null)}>
                  ✕
                </button>
              </div>
            )}
          </div>

          {selectedFile && !result && (
            <div className="upload-actions">
              <button 
                className="btn btn-primary btn-large"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="btn-spinner">⏳</span>
                    جاري التحليل... Analyzing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🔬</span>
                    بدء التحليل - Start Analysis
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="error-message">
              <span className="error-icon">⚠️</span>
              {error}
            </div>
          )}

          <div className="csv-requirements">
            <h4>⚙️ متطلبات ملف CSV / CSV Requirements</h4>
            <ul>
              <li>✅ يجب أن يحتوي الملف على الأعمدة التالية: hour, patient_id, age, gender, heart_rate, respiratory_rate, spo2_pct, temperature_c, systolic_bp, diastolic_bp, lactate, creatinine</li>
              <li>✅ File must contain columns: hour, patient_id, age, gender, heart_rate, respiratory_rate, spo2_pct, temperature_c, systolic_bp, diastolic_bp, lactate, creatinine</li>
              <li>✅ البيانات يجب أن تكون مرتبة حسب الساعة / Data should be sorted by hour</li>
            </ul>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <div className="prediction-result fade-in" ref={printRef}>
            {/* Print Header */}
            <div className="print-only print-header">
              <div className="print-logo">
                <h1>🏥 Hospital Patient Risk System</h1>
                <h2>نظام التنبؤ بتدهور حالة المرضى</h2>
              </div>
              <div className="print-info">
                <p><strong>Report Date:</strong> {new Date().toLocaleString('ar-PS')}</p>
                <p><strong>Patient ID:</strong> {result.patient_id}</p>
              </div>
            </div>

            <div className="result-header no-print">
              <h3>📊 نتائج التحليل</h3>
              <span className="patient-badge">
                <span className="badge-icon">👤</span>
                {result.patient_id}
              </span>
            </div>

            {/* Patient Info Card */}
            <div className="patient-info-card">
              <h4>📋 معلومات المريض / Patient Information</h4>
              <div className="info-grid">
                <div className="info-item">
                  <span className="info-label">رقم المريض - Patient ID</span>
                  <strong className="info-value">{result.patient_id}</strong>
                </div>
                <div className="info-item">
                  <span className="info-label">العمر - Age</span>
                  <strong className="info-value">{result.patient_age} years</strong>
                </div>
                <div className="info-item">
                  <span className="info-label">الجنس - Gender</span>
                  <strong className="info-value">{result.patient_gender}</strong>
                </div>
                <div className="info-item">
                  <span className="info-label">تاريخ التحليل - Analysis Date</span>
                  <strong className="info-value">{new Date(result.timestamp).toLocaleString('ar-PS')}</strong>
                </div>
              </div>
            </div>

            {/* Main Risk Indicator */}
            <div 
              className="risk-indicator"
              style={{ background: getRiskGradient(result.final_risk_score) }}
            >
              <div className="risk-icon-large">
                {getRiskLabel(result.final_risk_score).icon}
              </div>
              <h2 className="risk-category">
                {getRiskLabel(result.final_risk_score).ar}
                <span className="risk-category-en">{getRiskLabel(result.final_risk_score).en}</span>
              </h2>
              <div className="risk-score-display">
                {(result.final_risk_score * 100).toFixed(1)}%
              </div>
              <div className="risk-label">النتيجة النهائية / Final Risk Score</div>
            </div>

            {/* Deterioration Alert */}
            {hasSignificantDeterioration() && (
              <div className="alert-card deterioration-alert">
                <div className="alert-icon">⚠️</div>
                <div className="alert-content">
                  <h4>تنبيه تدهور الحالة / Deterioration Alert</h4>
                  <p>
                    تم اكتشاف ارتفاع كبير في نسبة الخطر عند الساعة <strong>{result.deterioration_hour}</strong>
                    <br />
                    Significant risk increase detected at hour <strong>{result.deterioration_hour}</strong>
                  </p>
                  <div className="alert-details">
                    <span>أعلى نسبة خطر: <strong>{(result.summary.max_risk * 100).toFixed(1)}%</strong></span>
                  </div>
                </div>
              </div>
            )}

            {/* Stable Alert */}
            {result.final_risk_score < 0.3 && result.summary.risk_trend !== 'increasing' && (
              <div className="alert-card success-alert">
                <div className="alert-icon">✅</div>
                <div className="alert-content">
                  <h4>حالة مستقرة / Stable Condition</h4>
                  <p>
                    الحالة الصحية للمريض مستقرة مع نسبة خطر منخفضة
                    <br />
                    Patient's health condition is stable with low risk level
                  </p>
                </div>
              </div>
            )}

            {/* Summary Statistics */}
            <div className="summary-card">
              <h4>📈 ملخص التحليل / Analysis Summary</h4>
              <div className="summary-grid">
                <div className="summary-item">
                  <div className="summary-icon">⏱️</div>
                  <span className="summary-label">إجمالي الساعات</span>
                  <strong className="summary-value">{result.total_hours}</strong>
                  <span className="summary-label-en">Total Hours</span>
                </div>
                <div className="summary-item">
                  <div className="summary-icon">📉</div>
                  <span className="summary-label">أدنى خطر</span>
                  <strong className="summary-value">{(result.summary.min_risk * 100).toFixed(1)}%</strong>
                  <span className="summary-label-en">Min Risk</span>
                </div>
                <div className="summary-item">
                  <div className="summary-icon">📊</div>
                  <span className="summary-label">متوسط الخطر</span>
                  <strong className="summary-value">{(result.summary.avg_risk * 100).toFixed(1)}%</strong>
                  <span className="summary-label-en">Avg Risk</span>
                </div>
                <div className="summary-item">
                  <div className="summary-icon">📈</div>
                  <span className="summary-label">أعلى خطر</span>
                  <strong className="summary-value">{(result.summary.max_risk * 100).toFixed(1)}%</strong>
                  <span className="summary-label-en">Max Risk</span>
                </div>
                <div className="summary-item">
                  <div className="summary-icon">{getTrendIcon(result.summary.risk_trend)}</div>
                  <span className="summary-label">الاتجاه</span>
                  <strong className="summary-value">
                    {getTrendLabel(result.summary.risk_trend).ar}
                  </strong>
                  <span className="summary-label-en">{getTrendLabel(result.summary.risk_trend).en}</span>
                </div>
                <div className="summary-item">
                  <div className="summary-icon">🔴</div>
                  <span className="summary-label">ساعات خطر عالي</span>
                  <strong className="summary-value">{result.summary.hours_high_risk}</strong>
                  <span className="summary-label-en">High Risk Hours</span>
                </div>
              </div>
            </div>

            {/* Hourly Timeline */}
            <div className="timeline-card">
              <h4>⏱️ الجدول الزمني للقراءات / Hourly Timeline</h4>
              <div className="timeline-container">
                {result.hourly_readings.map((reading, index) => {
                  const riskLabel = getRiskLabel(reading.risk_score);
                  const isHighRisk = reading.risk_score >= 0.5;
                  
                  return (
                    <div 
                      key={index} 
                      className={`timeline-item ${isHighRisk ? 'high-risk' : ''}`}
                    >
                      <div className="timeline-hour">
                        <span className="hour-label">Hour</span>
                        <strong className="hour-number">{reading.hour}</strong>
                      </div>
                      <div className="timeline-data">
                        <div className="timeline-risk-info">
                          <span className="risk-icon">{riskLabel.icon}</span>
                          <span className="risk-text">{riskLabel.en}</span>
                        </div>
                        <div 
                          className="timeline-risk-bar"
                          style={{
                            width: `${reading.risk_score * 100}%`,
                            background: getRiskColor(reading.risk_score)
                          }}
                        >
                          <span className="risk-value">
                            {(reading.risk_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      
                      {/* Detailed Vitals */}
                      <div className="vitals-detailed">
                        <div className="vitals-row">
                          <div className="vital-detail">
                            <span className="vital-label">SpO₂</span>
                            <span className="vital-value">{reading.spo2}%</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">HR</span>
                            <span className="vital-value">{reading.heart_rate}</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">RR</span>
                            <span className="vital-value">{reading.respiratory_rate}</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">Temp</span>
                            <span className="vital-value">{reading.temperature}°C</span>
                          </div>
                        </div>
                        <div className="vitals-row">
                          <div className="vital-detail">
                            <span className="vital-label">BP</span>
                            <span className="vital-value">{reading.systolic_bp}/{reading.diastolic_bp}</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">Lactate</span>
                            <span className="vital-value">{reading.lactate}</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">WBC</span>
                            <span className="vital-value">{reading.wbc}</span>
                          </div>
                          <div className="vital-detail">
                            <span className="vital-label">Creat</span>
                            <span className="vital-value">{reading.creatinine}</span>
                          </div>
                        </div>
                      </div>

                      {/* Risk Factors */}
                      {reading.risk_factors && reading.risk_factors.length > 0 && (
                        <div className="risk-factors">
                          <strong className="factors-label">Risk Factors:</strong>
                          <div className="factors-tags">
                            {reading.risk_factors.map((factor, idx) => (
                              <span key={idx} className="factor-tag">{factor}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Recommendations */}
            {result.recommendations && result.recommendations.length > 0 && (
              <div className="recommendations-card">
                <h4>💡 التوصيات الطبية / Clinical Recommendations</h4>
                <ul className="recommendations-list">
                  {result.recommendations.map((rec, index) => (
                    <li key={index}>
                      <span className="rec-bullet">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Print Footer */}
            <div className="print-only print-footer">
              <p>Generated by Hospital Patient Risk System</p>
              <p>This report is confidential and for medical use only</p>
            </div>

            {/* Action Buttons */}
            <div className="action-buttons no-print">
              <button onClick={handlePrint} className="btn btn-success btn-action">
                <span className="btn-icon">🖨️</span>
                طباعة التقرير - Print Report
              </button>
              <button onClick={handleReset} className="btn btn-secondary btn-action">
                <span className="btn-icon">📁</span>
                تحليل ملف آخر - New Analysis
              </button>
              <Link to="/" className="btn btn-primary btn-action">
                <span className="btn-icon">📊</span>
                عرض لوحة التحكم - Dashboard
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default CSVUpload;