import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Patientmanagement.css';


function PatientManagement() {
  const [patients, setPatients] = useState([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showReadingModal, setShowReadingModal] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [newPatient, setNewPatient] = useState({
    patient_id: '',
    name: '',
    age: '',
    gender: '0'
  });
  const [newReading, setNewReading] = useState({
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
    oxygen_flow: '0',
    oxygen_device: '0',
    nurse_alert: '0',
    mobility_score: '3',
    comorbidity_index: '0'
  });

  useEffect(() => {
    loadPatients();
  }, []);

  const loadPatients = () => {
    const savedPatients = JSON.parse(localStorage.getItem('activePatients') || '[]');
    setPatients(savedPatients);
  };

  const savePatients = (updatedPatients) => {
    localStorage.setItem('activePatients', JSON.stringify(updatedPatients));
    setPatients(updatedPatients);
  };

  const handleAddPatient = () => {
    if (!newPatient.patient_id || !newPatient.name || !newPatient.age) {
      alert('الرجاء ملء جميع الحقول الإلزامية / Please fill all required fields');
      return;
    }

    const patient = {
      ...newPatient,
      age: parseInt(newPatient.age),
      gender: parseInt(newPatient.gender),
      readings: [],
      created_at: new Date().toISOString(),
      last_updated: new Date().toISOString()
    };

    const updatedPatients = [...patients, patient];
    savePatients(updatedPatients);
    setShowAddModal(false);
    setNewPatient({ patient_id: '', name: '', age: '', gender: '0' });
    alert('تمت إضافة المريض بنجاح / Patient added successfully');
  };

  const calculateRiskScore = (reading) => {
    let riskScore = 0;

    // SpO2
    if (reading.spo2_pct < 90) riskScore += 0.25;
    else if (reading.spo2_pct < 92) riskScore += 0.15;
    else if (reading.spo2_pct < 95) riskScore += 0.08;

    // Heart Rate
    if (reading.heart_rate > 120 || reading.heart_rate < 50) riskScore += 0.20;
    else if (reading.heart_rate > 100 || reading.heart_rate < 60) riskScore += 0.10;

    // Respiratory Rate
    if (reading.respiratory_rate > 25 || reading.respiratory_rate < 10) riskScore += 0.20;
    else if (reading.respiratory_rate > 20) riskScore += 0.10;

    // Temperature
    if (reading.temperature_c > 38.5 || reading.temperature_c < 35.5) riskScore += 0.15;
    else if (reading.temperature_c > 37.8) riskScore += 0.08;

    // Blood Pressure
    if (reading.systolic_bp < 90 || reading.systolic_bp > 160) riskScore += 0.18;

    // Lactate
    if (reading.lactate > 4.0) riskScore += 0.25;
    else if (reading.lactate > 2.5) riskScore += 0.15;

    // WBC
    if (reading.wbc_count > 15 || reading.wbc_count < 4) riskScore += 0.15;

    // Creatinine
    if (reading.creatinine > 2.0) riskScore += 0.20;
    else if (reading.creatinine > 1.5) riskScore += 0.10;

    // CRP
    if (reading.crp_level > 100) riskScore += 0.20;
    else if (reading.crp_level > 50) riskScore += 0.12;

    // Hemoglobin
    if (reading.hemoglobin < 8 || reading.hemoglobin > 18) riskScore += 0.15;

    // Oxygen Support
    if (parseInt(reading.oxygen_device) >= 2) riskScore += 0.15;
    else if (parseFloat(reading.oxygen_flow) > 0) riskScore += 0.08;

    // Mobility
    if (parseInt(reading.mobility_score) <= 1) riskScore += 0.15;

    // Comorbidity
    if (parseInt(reading.comorbidity_index) >= 4) riskScore += 0.12;

    // Nurse Alert
    if (parseInt(reading.nurse_alert) === 1) riskScore += 0.10;

    return Math.min(riskScore, 1.0);
  };

  const handleAddReading = () => {
    if (!selectedPatient) return;

    // Validate required fields
    const requiredFields = ['heart_rate', 'respiratory_rate', 'spo2_pct', 'temperature_c', 
                           'systolic_bp', 'diastolic_bp', 'lactate', 'creatinine'];
    
    for (let field of requiredFields) {
      if (!newReading[field]) {
        alert('الرجاء ملء جميع الحقول الحيوية / Please fill all vital fields');
        return;
      }
    }

    const reading = {
      ...newReading,
      timestamp: new Date().toISOString(),
      hour: selectedPatient.readings.length,
      heart_rate: parseFloat(newReading.heart_rate),
      respiratory_rate: parseFloat(newReading.respiratory_rate),
      spo2_pct: parseFloat(newReading.spo2_pct),
      temperature_c: parseFloat(newReading.temperature_c),
      systolic_bp: parseFloat(newReading.systolic_bp),
      diastolic_bp: parseFloat(newReading.diastolic_bp),
      wbc_count: parseFloat(newReading.wbc_count) || 10,
      lactate: parseFloat(newReading.lactate),
      creatinine: parseFloat(newReading.creatinine),
      crp_level: parseFloat(newReading.crp_level) || 0,
      hemoglobin: parseFloat(newReading.hemoglobin) || 12,
      oxygen_flow: parseFloat(newReading.oxygen_flow),
      oxygen_device: parseInt(newReading.oxygen_device),
      nurse_alert: parseInt(newReading.nurse_alert),
      mobility_score: parseInt(newReading.mobility_score),
      comorbidity_index: parseInt(newReading.comorbidity_index)
    };

    reading.risk_score = calculateRiskScore(reading);

    const updatedPatients = patients.map(p => {
      if (p.patient_id === selectedPatient.patient_id) {
        return {
          ...p,
          readings: [...p.readings, reading],
          last_updated: new Date().toISOString(),
          current_risk: reading.risk_score
        };
      }
      return p;
    });

    savePatients(updatedPatients);
    setShowReadingModal(false);
    setNewReading({
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
      oxygen_flow: '0',
      oxygen_device: '0',
      nurse_alert: '0',
      mobility_score: '3',
      comorbidity_index: '0'
    });
    alert('تمت إضافة القراءة بنجاح / Reading added successfully');
  };

  const deletePatient = (patientId) => {
    if (window.confirm('هل أنت متأكد من حذف هذا المريض؟ / Are you sure you want to delete this patient?')) {
      const updatedPatients = patients.filter(p => p.patient_id !== patientId);
      savePatients(updatedPatients);
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
    if (score >= 0.7) return 'خطر عالي';
    if (score >= 0.5) return 'خطر متوسط';
    if (score >= 0.3) return 'خطر منخفض-متوسط';
    return 'خطر منخفض';
  };

  return (
    <div className="patient-management">
      <div className="page-header">
        <h2>👥 إدارة المرضى - Patient Management</h2>
        <p>إضافة ومتابعة المرضى وقراءاتهم الحيوية - Add and monitor patients and their vital readings</p>
      </div>

      <div className="management-controls">
        <button className="btn btn-primary" onClick={() => setShowAddModal(true)}>
          <span className="btn-icon">➕</span>
          إضافة مريض جديد - Add New Patient
        </button>
        <Link to="/" className="btn btn-secondary">
          <span className="btn-icon">📊</span>
          عرض الداشبورد - View Dashboard
        </Link>
      </div>

      {/* Patients Table */}
      {patients.length === 0 ? (
        <div className="no-data">
          <div className="no-data-icon">👤</div>
          <h3>لا يوجد مرضى - No Patients</h3>
          <p>ابدأ بإضافة مريض جديد - Start by adding a new patient</p>
        </div>
      ) : (
        <div className="patients-table-wrapper">
          <table className="patients-table">
            <thead>
              <tr>
                <th>الحالة<br/>Status</th>
                <th>رقم المريض<br/>Patient ID</th>
                <th>الاسم<br/>Name</th>
                <th>العمر<br/>Age</th>
                <th>الجنس<br/>Gender</th>
                <th>عدد القراءات<br/>Readings</th>
                <th>آخر تحديث<br/>Last Update</th>
                <th>الخطر الحالي<br/>Current Risk</th>
                <th>إجراءات<br/>Actions</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((patient, index) => (
                <tr key={index}>
                  <td>
                    <div 
                      className="status-indicator"
                      style={{ backgroundColor: getRiskColor(patient.current_risk) }}
                      title={getRiskLabel(patient.current_risk)}
                    ></div>
                  </td>
                  <td className="patient-id-cell">
                    <strong>{patient.patient_id}</strong>
                  </td>
                  <td>{patient.name}</td>
                  <td>{patient.age}</td>
                  <td>{patient.gender === 1 ? 'ذكر - Male' : 'أنثى - Female'}</td>
                  <td>
                    <span className="readings-badge">
                      {patient.readings.length} قراءة
                    </span>
                  </td>
                  <td className="date-cell">
                    {new Date(patient.last_updated).toLocaleString('ar-PS')}
                  </td>
                  <td>
                    {patient.current_risk ? (
                      <span 
                        className="risk-badge"
                        style={{ backgroundColor: getRiskColor(patient.current_risk) }}
                      >
                        {(patient.current_risk * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="no-risk">-</span>
                    )}
                  </td>
                  <td className="actions-cell">
                    <button 
                      className="btn-action btn-add-reading"
                      onClick={() => {
                        setSelectedPatient(patient);
                        setShowReadingModal(true);
                      }}
                      title="إضافة قراءة"
                    >
                      ➕
                    </button>
                    <Link 
                      to={`/patient/${patient.patient_id}`}
                      className="btn-action btn-view"
                      title="عرض التفاصيل"
                    >
                      👁️
                    </Link>
                    <button 
                      className="btn-action btn-delete"
                      onClick={() => deletePatient(patient.patient_id)}
                      title="حذف"
                    >
                      🗑️
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Add Patient Modal */}
      {showAddModal && (
        <div className="modal-overlay" onClick={() => setShowAddModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>➕ إضافة مريض جديد - Add New Patient</h3>
              <button className="modal-close" onClick={() => setShowAddModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="form-grid">
                <div className="form-group">
                  <label>رقم المريض - Patient ID *</label>
                  <input
                    type="text"
                    value={newPatient.patient_id}
                    onChange={(e) => setNewPatient({...newPatient, patient_id: e.target.value})}
                    placeholder="P001"
                  />
                </div>
                <div className="form-group">
                  <label>الاسم - Name *</label>
                  <input
                    type="text"
                    value={newPatient.name}
                    onChange={(e) => setNewPatient({...newPatient, name: e.target.value})}
                    placeholder="اسم المريض"
                  />
                </div>
                <div className="form-group">
                  <label>العمر - Age *</label>
                  <input
                    type="number"
                    value={newPatient.age}
                    onChange={(e) => setNewPatient({...newPatient, age: e.target.value})}
                    placeholder="50"
                  />
                </div>
                <div className="form-group">
                  <label>الجنس - Gender *</label>
                  <select
                    value={newPatient.gender}
                    onChange={(e) => setNewPatient({...newPatient, gender: e.target.value})}
                  >
                    <option value="0">أنثى - Female</option>
                    <option value="1">ذكر - Male</option>
                  </select>
                </div>
              </div>
              <div className="modal-actions">
                <button className="btn btn-primary" onClick={handleAddPatient}>
                  ✅ إضافة - Add
                </button>
                <button className="btn btn-secondary" onClick={() => setShowAddModal(false)}>
                  ✕ إلغاء - Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Add Reading Modal */}
      {showReadingModal && selectedPatient && (
        <div className="modal-overlay" onClick={() => setShowReadingModal(false)}>
          <div className="modal-content modal-large" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📊 إضافة قراءة جديدة - Add New Reading</h3>
              <div className="patient-info-small">
                <span>👤 {selectedPatient.name}</span>
                <span>🆔 {selectedPatient.patient_id}</span>
              </div>
              <button className="modal-close" onClick={() => setShowReadingModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="reading-form">
                <div className="form-section-header">
                  <h4>🫀 العلامات الحيوية - Vital Signs</h4>
                </div>
                <div className="form-grid">
                  <div className="form-group">
                    <label>معدل نبضات القلب - Heart Rate (bpm) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.heart_rate}
                      onChange={(e) => setNewReading({...newReading, heart_rate: e.target.value})}
                      placeholder="75"
                    />
                  </div>
                  <div className="form-group">
                    <label>معدل التنفس - Respiratory Rate (breaths/min) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.respiratory_rate}
                      onChange={(e) => setNewReading({...newReading, respiratory_rate: e.target.value})}
                      placeholder="16"
                    />
                  </div>
                  <div className="form-group">
                    <label>نسبة الأكسجين - SpO2 (%) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.spo2_pct}
                      onChange={(e) => setNewReading({...newReading, spo2_pct: e.target.value})}
                      placeholder="98"
                    />
                  </div>
                  <div className="form-group">
                    <label>درجة الحرارة - Temperature (°C) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.temperature_c}
                      onChange={(e) => setNewReading({...newReading, temperature_c: e.target.value})}
                      placeholder="37.0"
                    />
                  </div>
                  <div className="form-group">
                    <label>ضغط الدم الانقباضي - Systolic BP (mmHg) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.systolic_bp}
                      onChange={(e) => setNewReading({...newReading, systolic_bp: e.target.value})}
                      placeholder="120"
                    />
                  </div>
                  <div className="form-group">
                    <label>ضغط الدم الانبساطي - Diastolic BP (mmHg) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.diastolic_bp}
                      onChange={(e) => setNewReading({...newReading, diastolic_bp: e.target.value})}
                      placeholder="80"
                    />
                  </div>
                </div>

                <div className="form-section-header">
                  <h4>🔬 التحاليل المخبرية - Laboratory Tests</h4>
                </div>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Lactate (mmol/L) *</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.lactate}
                      onChange={(e) => setNewReading({...newReading, lactate: e.target.value})}
                      placeholder="1.5"
                    />
                  </div>
                  <div className="form-group">
                    <label>Creatinine (mg/dL) *</label>
                    <input
                      type="number"
                      step="0.01"
                      value={newReading.creatinine}
                      onChange={(e) => setNewReading({...newReading, creatinine: e.target.value})}
                      placeholder="1.0"
                    />
                  </div>
                  <div className="form-group">
                    <label>WBC Count (×10³/μL)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.wbc_count}
                      onChange={(e) => setNewReading({...newReading, wbc_count: e.target.value})}
                      placeholder="10.0"
                    />
                  </div>
                  <div className="form-group">
                    <label>CRP Level (mg/L)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.crp_level}
                      onChange={(e) => setNewReading({...newReading, crp_level: e.target.value})}
                      placeholder="5.0"
                    />
                  </div>
                  <div className="form-group">
                    <label>Hemoglobin (g/dL)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={newReading.hemoglobin}
                      onChange={(e) => setNewReading({...newReading, hemoglobin: e.target.value})}
                      placeholder="14.0"
                    />
                  </div>
                </div>

                <div className="form-section-header">
                  <h4>⚙️ معلومات إضافية - Additional Information</h4>
                </div>
                <div className="form-grid">
                  <div className="form-group">
                    <label>Oxygen Flow (L/min)</label>
                    <input
                      type="number"
                      step="0.5"
                      value={newReading.oxygen_flow}
                      onChange={(e) => setNewReading({...newReading, oxygen_flow: e.target.value})}
                      placeholder="0"
                    />
                  </div>
                  <div className="form-group">
                    <label>Oxygen Device</label>
                    <select
                      value={newReading.oxygen_device}
                      onChange={(e) => setNewReading({...newReading, oxygen_device: e.target.value})}
                    >
                      <option value="0">None</option>
                      <option value="1">Nasal Cannula</option>
                      <option value="2">Mask</option>
                      <option value="3">High Flow</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Mobility Score</label>
                    <select
                      value={newReading.mobility_score}
                      onChange={(e) => setNewReading({...newReading, mobility_score: e.target.value})}
                    >
                      <option value="0">Immobile</option>
                      <option value="1">Limited</option>
                      <option value="2">Moderate</option>
                      <option value="3">Good</option>
                      <option value="4">Excellent</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Comorbidity Index</label>
                    <input
                      type="number"
                      value={newReading.comorbidity_index}
                      onChange={(e) => setNewReading({...newReading, comorbidity_index: e.target.value})}
                      placeholder="0"
                    />
                  </div>
                  <div className="form-group">
                    <label>Nurse Alert</label>
                    <select
                      value={newReading.nurse_alert}
                      onChange={(e) => setNewReading({...newReading, nurse_alert: e.target.value})}
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="modal-actions">
                <button className="btn btn-primary" onClick={handleAddReading}>
                  ✅ إضافة القراءة - Add Reading
                </button>
                <button className="btn btn-secondary" onClick={() => setShowReadingModal(false)}>
                  ✕ إلغاء - Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PatientManagement;