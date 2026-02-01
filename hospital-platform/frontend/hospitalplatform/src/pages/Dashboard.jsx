// Dashboard.js
import React, { useState, useEffect } from 'react';
import './Dashboard.css';

function Dashboard() {
  const [modelStatus, setModelStatus] = useState(null);
  const [stats, setStats] = useState({
    totalPredictions: 0,
    highRiskPatients: 0,
    mediumRiskPatients: 0,
    lowRiskPatients: 0
  });
  const [patients, setPatients] = useState([]);
  const [filteredPatients, setFilteredPatients] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [filterRisk, setFilterRisk] = useState('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    applyFilter();
  }, [filterRisk, patients]);

  const fetchDashboardData = async () => {
    try {
      const statusResponse = await fetch('http://localhost:8000/model/status');
      const statusData = await statusResponse.json();
      setModelStatus(statusData);

      const savedPredictions = JSON.parse(
        localStorage.getItem('predictionHistory') || '[]'
      );

      const total = savedPredictions.length;
      const highRisk = savedPredictions.filter(p => p.risk_score >= 0.7).length;
      const mediumRisk = savedPredictions.filter(
        p => p.risk_score >= 0.4 && p.risk_score < 0.7
      ).length;
      const lowRisk = savedPredictions.filter(p => p.risk_score < 0.4).length;

      setStats({
        totalPredictions: total,
        highRiskPatients: highRisk,
        mediumRiskPatients: mediumRisk,
        lowRiskPatients: lowRisk
      });

      setPatients(savedPredictions);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const applyFilter = () => {
    if (filterRisk === 'all') {
      setFilteredPatients(patients);
    } else {
      setFilteredPatients(
        patients.filter(p => {
          if (filterRisk === 'high') return p.risk_score >= 0.7;
          if (filterRisk === 'medium')
            return p.risk_score >= 0.4 && p.risk_score < 0.7;
          if (filterRisk === 'low') return p.risk_score < 0.4;
          return true;
        })
      );
    }
  };

  const getRiskColor = score => {
    if (score >= 0.7) return '#ef4444';
    if (score >= 0.4) return '#f59e0b';
    return '#10b981';
  };

  const getRiskLabel = score => {
    if (score >= 0.7) return 'خطر عالي - High Risk';
    if (score >= 0.4) return 'خطر متوسط - Medium Risk';
    return 'خطر منخفض - Low Risk';
  };

  const getRiskIcon = score => {
    if (score >= 0.7) return '🚨';
    if (score >= 0.4) return '⚠️';
    return '✅';
  };

  const formatDate = dateString =>
    new Date(dateString).toLocaleString('ar-PS', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });

  if (loading) {
    return (
      <div className="dashboard">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>جاري تحميل البيانات...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="page-header">
        <h2>🏥 لوحة التحكم - Dashboard</h2>
        <p>المتابعة الكاملة لحالات المرضى من مكان واحد</p>
      </div>

      {/* Model Status */}
      <div className="status-card">
        <div className="status-header">
          <h3>🤖 حالة النموذج</h3>
          {modelStatus && (
            <span
              className={`status-indicator ${
                modelStatus.trained ? 'active' : 'inactive'
              }`}
            >
              {modelStatus.trained ? '● جاهز' : '○ غير مدرب'}
            </span>
          )}
        </div>

        {modelStatus?.trained && (
          <div className="status-details">
            <div className="status-item">
              <span>نوع النموذج:</span>
              <span>{modelStatus.model_type}</span>
            </div>
            <div className="status-item">
              <span>عدد الميزات:</span>
              <span>{modelStatus.features_count}</span>
            </div>
            <div className="status-item">
              <span>آخر تحديث:</span>
              <span>
                {modelStatus.last_updated
                  ? formatDate(modelStatus.last_updated)
                  : 'N/A'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card total">
          <div className="stat-value">{stats.totalPredictions}</div>
          <div className="stat-label">إجمالي المرضى</div>
        </div>
        <div className="stat-card high-risk">
          <div className="stat-value">{stats.highRiskPatients}</div>
          <div className="stat-label">خطر عالي</div>
        </div>
        <div className="stat-card medium-risk">
          <div className="stat-value">{stats.mediumRiskPatients}</div>
          <div className="stat-label">خطر متوسط</div>
        </div>
        <div className="stat-card low-risk">
          <div className="stat-value">{stats.lowRiskPatients}</div>
          <div className="stat-label">خطر منخفض</div>
        </div>
      </div>

      {/* Patients */}
      <div className="patients-section">
        <h3>👥 المرضى (المتابعة من هنا)</h3>

        <div className="filter-buttons">
          {['all', 'high', 'medium', 'low'].map(type => (
            <button
              key={type}
              className={filterRisk === type ? 'active' : ''}
              onClick={() => setFilterRisk(type)}
            >
              {type}
            </button>
          ))}
        </div>

        <div className="patient-cards-grid">
          {filteredPatients.map((patient, index) => (
            <div
              key={index}
              className="patient-card"
              style={{
                borderLeft: `5px solid ${getRiskColor(patient.risk_score)}`
              }}
              onClick={() => setSelectedPatient(patient)} // MODIFIED: المتابعة من الداشبورد
            >
              <div className="card-header">
                <strong>{patient.patient_id}</strong>
                <span>
                  {getRiskIcon(patient.risk_score)}{' '}
                  {(patient.risk_score * 100).toFixed(1)}%
                </span>
              </div>

              <div className="card-body">
                <p>{getRiskLabel(patient.risk_score)}</p>
                <p>
                  {patient.prediction === 1
                    ? '⚠️ تدهور متوقع'
                    : '✅ مستقر'}
                </p>
              </div>

              <div className="card-footer">
                {formatDate(patient.timestamp)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Patient Modal (بقي كما هو) */}
      {selectedPatient && (
        <div
          className="modal-overlay"
          onClick={() => setSelectedPatient(null)}
        >
          <div
            className="modal-content"
            onClick={e => e.stopPropagation()}
          >
            <div className="modal-header">
              <h3>📋 ملف المريض</h3>
              <button onClick={() => setSelectedPatient(null)}>✕</button>
            </div>

            <div className="modal-body">
              <p>
                <strong>Patient ID:</strong>{' '}
                {selectedPatient.patient_id}
              </p>
              <p>
                <strong>Risk:</strong>{' '}
                {(selectedPatient.risk_score * 100).toFixed(1)}%
              </p>
              <p>
                <strong>Prediction:</strong>{' '}
                {selectedPatient.prediction === 1
                  ? 'Deterioration'
                  : 'Stable'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
