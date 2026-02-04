import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import './Patientdetails.css';


function PatientDetails() {
  const { patientId } = useParams();
  const navigate = useNavigate();
  const [patient, setPatient] = useState(null);
  const [editingReading, setEditingReading] = useState(null);
  const [editForm, setEditForm] = useState({});

  useEffect(() => {
    loadPatient();
  }, [patientId]);

  const loadPatient = () => {
    const patients = JSON.parse(localStorage.getItem('activePatients') || '[]');
    const foundPatient = patients.find(p => p.patient_id === patientId);
    if (foundPatient) {
      setPatient(foundPatient);
    } else {
      alert('المريض غير موجود / Patient not found');
      navigate('/patients');
    }
  };

  const getRiskColor = (score) => {
    if (!score) return '#9ca3af';
    if (score >= 0.7) return '#ef4444';
    if (score >= 0.5) return '#f59e0b';
    if (score >= 0.3) return '#eab308';
    return '#10b981';
  };

  const getRiskLabel = (score) => {
    if (!score) return 'لا توجد قراءات';
    if (score >= 0.7) return 'خطر عالي - High Risk';
    if (score >= 0.5) return 'خطر متوسط - Medium Risk';
    if (score >= 0.3) return 'خطر منخفض-متوسط - Low-Medium Risk';
    return 'خطر منخفض - Low Risk';
  };

  const getStats = () => {
    if (!patient || patient.readings.length === 0) {
      return { totalReadings: 0, avgRisk: 0, maxRisk: 0, minRisk: 0, trend: 'stable' };
    }

    const risks = patient.readings.map(r => r.risk_score);
    const avgRisk = risks.reduce((a, b) => a + b, 0) / risks.length;
    const maxRisk = Math.max(...risks);
    const minRisk = Math.min(...risks);

    // Calculate trend
    const firstHalf = risks.slice(0, Math.floor(risks.length / 2));
    const secondHalf = risks.slice(Math.floor(risks.length / 2));
    const firstAvg = firstHalf.length > 0 ? firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length : 0;
    const secondAvg = secondHalf.length > 0 ? secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length : 0;

    let trend = 'stable';
    if (secondAvg > firstAvg + 0.1) trend = 'increasing';
    else if (secondAvg < firstAvg - 0.1) trend = 'decreasing';

    return {
      totalReadings: patient.readings.length,
      avgRisk,
      maxRisk,
      minRisk,
      trend
    };
  };

  const deleteReading = (index) => {
    if (window.confirm('هل أنت متأكد من حذف هذه القراءة؟ / Are you sure you want to delete this reading?')) {
      const updatedReadings = patient.readings.filter((_, i) => i !== index);
      const updatedPatient = {
        ...patient,
        readings: updatedReadings,
        current_risk: updatedReadings.length > 0 ? updatedReadings[updatedReadings.length - 1].risk_score : null,
        last_updated: new Date().toISOString()
      };

      const patients = JSON.parse(localStorage.getItem('activePatients') || '[]');
      const updatedPatients = patients.map(p => 
        p.patient_id === patientId ? updatedPatient : p
      );
      localStorage.setItem('activePatients', JSON.stringify(updatedPatients));
      setPatient(updatedPatient);
    }
  };

  const startEdit = (reading, index) => {
    setEditingReading(index);
    setEditForm(reading);
  };

  const saveEdit = () => {
    const updatedReadings = [...patient.readings];
    updatedReadings[editingReading] = editForm;

    const updatedPatient = {
      ...patient,
      readings: updatedReadings,
      current_risk: updatedReadings[updatedReadings.length - 1].risk_score,
      last_updated: new Date().toISOString()
    };

    const patients = JSON.parse(localStorage.getItem('activePatients') || '[]');
    const updatedPatients = patients.map(p => 
      p.patient_id === patientId ? updatedPatient : p
    );
    localStorage.setItem('activePatients', JSON.stringify(updatedPatients));
    setPatient(updatedPatient);
    setEditingReading(null);
    setEditForm({});
  };

  const handlePrint = () => {
    window.print();
  };

  const getTrendIcon = (trend) => {
    if (trend === 'increasing') return '📈';
    if (trend === 'decreasing') return '📉';
    return '➡️';
  };

  const getTrendLabel = (trend) => {
    if (trend === 'increasing') return 'متزايد - Increasing';
    if (trend === 'decreasing') return 'متناقص - Decreasing';
    return 'مستقر - Stable';
  };

  if (!patient) {
    return (
      <div className="patient-details">
        <div className="loading">جاري التحميل... Loading...</div>
      </div>
    );
  }

  const stats = getStats();

  return (
    <div className="patient-details">
      {/* Print Header */}
      <div className="print-only print-header">
        <div className="print-logo">
          <h1>🏥 Hospital Patient Risk System</h1>
          <h2>تقرير متابعة المريض - Patient Monitoring Report</h2>
        </div>
        <div className="print-info">
          <p><strong>Report Date:</strong> {new Date().toLocaleString('ar-PS')}</p>
          <p><strong>Patient ID:</strong> {patient.patient_id}</p>
        </div>
      </div>

      {/* Header */}
      <div className="details-header no-print">
        <div className="header-content">
          <h2>📋 ملف المريض - Patient File</h2>
          <div className="header-actions">
            <button className="btn btn-print" onClick={handlePrint}>
              🖨️ طباعة
            </button>
            <Link to="/patients" className="btn btn-secondary">
              ← رجوع
            </Link>
          </div>
        </div>
      </div>

      {/* Patient Info Card */}
      <div className="patient-info-card">
        <div className="info-header">
          <div className="info-title">
            <h3>👤 معلومات المريض - Patient Information</h3>
          </div>
          <div 
            className="current-status"
            style={{ backgroundColor: getRiskColor(patient.current_risk) }}
          >
            <span className="status-label">الحالة الحالية</span>
            <span className="status-value">
              {patient.current_risk ? `${(patient.current_risk * 100).toFixed(1)}%` : 'N/A'}
            </span>
          </div>
        </div>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">رقم المريض - ID</span>
            <strong className="info-value">{patient.patient_id}</strong>
          </div>
          <div className="info-item">
            <span className="info-label">الاسم - Name</span>
            <strong className="info-value">{patient.name}</strong>
          </div>
          <div className="info-item">
            <span className="info-label">العمر - Age</span>
            <strong className="info-value">{patient.age} سنة</strong>
          </div>
          <div className="info-item">
            <span className="info-label">الجنس - Gender</span>
            <strong className="info-value">{patient.gender === 1 ? 'ذكر - Male' : 'أنثى - Female'}</strong>
          </div>
          <div className="info-item">
            <span className="info-label">تاريخ الإضافة - Created</span>
            <strong className="info-value">{new Date(patient.created_at).toLocaleDateString('ar-PS')}</strong>
          </div>
          <div className="info-item">
            <span className="info-label">آخر تحديث - Last Update</span>
            <strong className="info-value">{new Date(patient.last_updated).toLocaleString('ar-PS')}</strong>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      {patient.readings.length > 0 && (
        <div className="stats-cards">
          <div className="stat-card">
            <div className="stat-icon">📊</div>
            <div className="stat-info">
              <div className="stat-value">{stats.totalReadings}</div>
              <div className="stat-label">عدد القراءات<br/>Total Readings</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📉</div>
            <div className="stat-info">
              <div className="stat-value">{(stats.minRisk * 100).toFixed(1)}%</div>
              <div className="stat-label">أدنى خطر<br/>Min Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">📈</div>
            <div className="stat-info">
              <div className="stat-value">{(stats.avgRisk * 100).toFixed(1)}%</div>
              <div className="stat-label">متوسط الخطر<br/>Avg Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">🔴</div>
            <div className="stat-info">
              <div className="stat-value">{(stats.maxRisk * 100).toFixed(1)}%</div>
              <div className="stat-label">أعلى خطر<br/>Max Risk</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon">{getTrendIcon(stats.trend)}</div>
            <div className="stat-info">
              <div className="stat-value-small">{getTrendLabel(stats.trend).split(' - ')[0]}</div>
              <div className="stat-label">الاتجاه<br/>Trend</div>
            </div>
          </div>
        </div>
      )}

      {/* Readings Timeline */}
      <div className="readings-section">
        <div className="section-header">
          <h3>⏱️ سجل القراءات - Readings History</h3>
          <Link to="/patients" className="btn btn-primary no-print">
            ➕ إضافة قراءة جديدة
          </Link>
        </div>

        {patient.readings.length === 0 ? (
          <div className="no-readings">
            <div className="no-readings-icon">📭</div>
            <h3>لا توجد قراءات - No Readings</h3>
            <p>ابدأ بإضافة قراءات لهذا المريض - Start by adding readings for this patient</p>
          </div>
        ) : (
          <div className="readings-timeline">
            {patient.readings.map((reading, index) => (
              <div 
                key={index} 
                className={`reading-card ${reading.risk_score >= 0.5 ? 'high-risk' : ''}`}
              >
                <div className="reading-header">
                  <div className="reading-time">
                    <span className="reading-hour">القراءة {index + 1}</span>
                    <span className="reading-date">
                      {new Date(reading.timestamp).toLocaleString('ar-PS')}
                    </span>
                  </div>
                  <div className="reading-risk">
                    <div 
                      className="risk-circle"
                      style={{ backgroundColor: getRiskColor(reading.risk_score) }}
                    >
                      {(reading.risk_score * 100).toFixed(1)}%
                    </div>
                    <span className="risk-text">{getRiskLabel(reading.risk_score)}</span>
                  </div>
                  <div className="reading-actions no-print">
                    {editingReading === index ? (
                      <>
                        <button className="btn-action btn-save" onClick={saveEdit}>
                          ✅
                        </button>
                        <button className="btn-action btn-cancel" onClick={() => setEditingReading(null)}>
                          ✕
                        </button>
                      </>
                    ) : (
                      <>
                        <button className="btn-action btn-edit" onClick={() => startEdit(reading, index)}>
                          ✏️
                        </button>
                        <button className="btn-action btn-delete" onClick={() => deleteReading(index)}>
                          🗑️
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {editingReading === index ? (
                  <div className="edit-form">
                    <div className="edit-grid">
                      <div className="edit-field">
                        <label>Heart Rate</label>
                        <input
                          type="number"
                          value={editForm.heart_rate}
                          onChange={(e) => setEditForm({...editForm, heart_rate: parseFloat(e.target.value)})}
                        />
                      </div>
                      <div className="edit-field">
                        <label>SpO2 (%)</label>
                        <input
                          type="number"
                          value={editForm.spo2_pct}
                          onChange={(e) => setEditForm({...editForm, spo2_pct: parseFloat(e.target.value)})}
                        />
                      </div>
                      <div className="edit-field">
                        <label>Temp (°C)</label>
                        <input
                          type="number"
                          value={editForm.temperature_c}
                          onChange={(e) => setEditForm({...editForm, temperature_c: parseFloat(e.target.value)})}
                        />
                      </div>
                      <div className="edit-field">
                        <label>Systolic BP</label>
                        <input
                          type="number"
                          value={editForm.systolic_bp}
                          onChange={(e) => setEditForm({...editForm, systolic_bp: parseFloat(e.target.value)})}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="reading-vitals">
                    <div className="vitals-grid">
                      <div className="vital-item">
                        <span className="vital-icon">💓</span>
                        <span className="vital-label">HR</span>
                        <span className="vital-value">{reading.heart_rate}</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">🫁</span>
                        <span className="vital-label">RR</span>
                        <span className="vital-value">{reading.respiratory_rate}</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">💨</span>
                        <span className="vital-label">SpO2</span>
                        <span className="vital-value">{reading.spo2_pct}%</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">🌡️</span>
                        <span className="vital-label">Temp</span>
                        <span className="vital-value">{reading.temperature_c}°C</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">🩸</span>
                        <span className="vital-label">BP</span>
                        <span className="vital-value">{reading.systolic_bp}/{reading.diastolic_bp}</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">🧪</span>
                        <span className="vital-label">Lactate</span>
                        <span className="vital-value">{reading.lactate}</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">🔬</span>
                        <span className="vital-label">Creat</span>
                        <span className="vital-value">{reading.creatinine}</span>
                      </div>
                      <div className="vital-item">
                        <span className="vital-icon">⚪</span>
                        <span className="vital-label">WBC</span>
                        <span className="vital-value">{reading.wbc_count}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Print Footer */}
      <div className="print-only print-footer">
        <p>Generated by Hospital Patient Risk System</p>
        <p>This report is confidential and for medical use only</p>
      </div>
    </div>
  );
}

export default PatientDetails;