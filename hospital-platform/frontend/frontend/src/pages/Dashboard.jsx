// Dashboard.js
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Dashboard.css';

function Dashboard() {
  const [predictions, setPredictions] = useState([]);
  const [filteredPredictions, setFilteredPredictions] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('all');
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [sortBy, setSortBy] = useState('date-desc');

  useEffect(() => {
    loadPredictions();
    // Refresh data every 5 seconds to catch new predictions
    const interval = setInterval(loadPredictions, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    filterAndSortPredictions();
  }, [searchTerm, filterRisk, sortBy, predictions]);

  const loadPredictions = () => {
    const savedPredictions = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
    setPredictions(savedPredictions);
  };

  const filterAndSortPredictions = () => {
    let filtered = [...predictions];

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(p => 
        p.patient_id?.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Risk filter
    if (filterRisk !== 'all') {
      filtered = filtered.filter(p => {
        const score = p.risk_score * 100;
        if (filterRisk === 'high') return score >= 70;
        if (filterRisk === 'medium') return score >= 40 && score < 70;
        if (filterRisk === 'low') return score < 40;
        return true;
      });
    }

    // Sort
    filtered.sort((a, b) => {
      switch(sortBy) {
        case 'date-desc':
          return new Date(b.timestamp) - new Date(a.timestamp);
        case 'date-asc':
          return new Date(a.timestamp) - new Date(b.timestamp);
        case 'risk-desc':
          return (b.risk_score * 100) - (a.risk_score * 100);
        case 'risk-asc':
          return (a.risk_score * 100) - (b.risk_score * 100);
        default:
          return 0;
      }
    });

    setFilteredPredictions(filtered);
  };

  const getRiskColor = (score) => {
    const percentage = score * 100;
    if (percentage >= 70) return '#ef4444';
    if (percentage >= 40) return '#f59e0b';
    return '#10b981';
  };

  const getRiskCategory = (score) => {
    const percentage = score * 100;
    if (percentage >= 70) return 'خطر عالي - High Risk';
    if (percentage >= 40) return 'خطر متوسط - Medium Risk';
    return 'خطر منخفض - Low Risk';
  };

  const clearHistory = () => {
    if (window.confirm('هل أنت متأكد من حذف جميع السجلات؟ Are you sure you want to clear all history?')) {
      localStorage.removeItem('predictionHistory');
      setPredictions([]);
      setFilteredPredictions([]);
      setSelectedPrediction(null);
    }
  };

  const exportData = () => {
    const dataStr = JSON.stringify(predictions, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `patient-history-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const exportCSV = () => {
    if (predictions.length === 0) return;

    const headers = ['Patient ID', 'Risk Score', 'Risk Category', 'Prediction', 'Confidence', 'Timestamp'];
    const rows = predictions.map(p => [
      p.patient_id,
      (p.risk_score * 100).toFixed(1) + '%',
      getRiskCategory(p.risk_score),
      p.prediction === 1 ? 'Deterioration' : 'Stable',
      (p.confidence * 100).toFixed(1) + '%',
      new Date(p.timestamp).toLocaleString()
    ]);

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `patient-history-${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const printPrediction = (pred) => {
    setSelectedPrediction(pred);
    setTimeout(() => {
      window.print();
    }, 100);
  };

  const getStats = () => {
    const total = predictions.length;
    const high = predictions.filter(p => p.risk_score * 100 >= 70).length;
    const medium = predictions.filter(p => {
      const score = p.risk_score * 100;
      return score >= 40 && score < 70;
    }).length;
    const low = predictions.filter(p => p.risk_score * 100 < 40).length;

    return { total, high, medium, low };
  };

  const stats = getStats();

  return (
    <div className="patient-history">
      <div className="page-header no-print">
        <h2>📚 سجل المرضى - Patient History</h2>
        <p>عرض جميع التنبؤات السابقة - View all past predictions and assessments</p>
        <div className="header-quick-links">
          <Link to="/patients" className="quick-link">
            👥 إدارة المرضى - Manage Patients
          </Link>
          <Link to="/predict" className="quick-link">
            📊 تحليل CSV - CSV Analysis
          </Link>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="stats-cards no-print">
        <div className="stat-card total-card">
          <div className="stat-icon">📊</div>
          <div className="stat-info">
            <div className="stat-value">{stats.total}</div>
            <div className="stat-label">إجمالي السجلات<br/>Total Records</div>
          </div>
        </div>
        
        <div className="stat-card high-card">
          <div className="stat-icon">🚨</div>
          <div className="stat-info">
            <div className="stat-value">{stats.high}</div>
            <div className="stat-label">خطر عالي<br/>High Risk</div>
          </div>
        </div>
        
        <div className="stat-card medium-card">
          <div className="stat-icon">⚠️</div>
          <div className="stat-info">
            <div className="stat-value">{stats.medium}</div>
            <div className="stat-label">خطر متوسط<br/>Medium Risk</div>
          </div>
        </div>
        
        <div className="stat-card low-card">
          <div className="stat-icon">✅</div>
          <div className="stat-info">
            <div className="stat-value">{stats.low}</div>
            <div className="stat-label">خطر منخفض<br/>Low Risk</div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="history-controls no-print">
        <div className="search-section">
          <div className="search-input">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              placeholder="البحث برقم المريض - Search by Patient ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        <div className="filter-section">
          <select 
            value={filterRisk} 
            onChange={(e) => setFilterRisk(e.target.value)}
            className="filter-select"
          >
            <option value="all">جميع المستويات - All Levels</option>
            <option value="high">خطر عالي - High Risk</option>
            <option value="medium">خطر متوسط - Medium Risk</option>
            <option value="low">خطر منخفض - Low Risk</option>
          </select>

          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="filter-select"
          >
            <option value="date-desc">الأحدث أولاً - Newest First</option>
            <option value="date-asc">الأقدم أولاً - Oldest First</option>
            <option value="risk-desc">الخطر (عالي-منخفض) - Risk (High-Low)</option>
            <option value="risk-asc">الخطر (منخفض-عالي) - Risk (Low-High)</option>
          </select>
        </div>

        <div className="action-section">
          <button className="btn-secondary" onClick={exportCSV}>
            📄 CSV تصدير
          </button>
          <button className="btn-secondary" onClick={exportData}>
            📥 JSON تصدير
          </button>
          <button className="btn-danger" onClick={clearHistory}>
            🗑️ مسح الكل
          </button>
        </div>
      </div>

      {/* Results */}
      {filteredPredictions.length === 0 ? (
        <div className="no-data no-print">
          <div className="no-data-icon">📭</div>
          <h3>لا توجد نتائج - No Results Found</h3>
          <p>
            {predictions.length === 0 
              ? 'ابدأ بإضافة تنبؤات لرؤية السجل - Start making predictions to see history here'
              : 'جرب تعديل معايير البحث أو التصفية - Try adjusting your search or filter criteria'
            }
          </p>
        </div>
      ) : (
        <div className="history-table-wrapper">
          <div className="table-container">
            <table className="history-table">
              <thead>
                <tr>
                  <th>رقم المريض<br/>Patient ID</th>
                  <th>درجة الخطر<br/>Risk Score</th>
                  <th>التصنيف<br/>Category</th>
                  <th>التنبؤ<br/>Prediction</th>
                  <th>الثقة<br/>Confidence</th>
                  <th>التاريخ<br/>Date</th>
                  <th className="no-print">إجراء<br/>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((pred, index) => (
                  <tr key={index} className="table-row">
                    <td className="patient-id-cell">
                      <strong>{pred.patient_id}</strong>
                    </td>
                    <td className="risk-score-cell">
                      <div 
                        className="risk-score-badge"
                        style={{ backgroundColor: getRiskColor(pred.risk_score) }}
                      >
                        {(pred.risk_score * 100).toFixed(1)}%
                      </div>
                    </td>
                    <td>
                      <span className="risk-category-badge">
                        {getRiskCategory(pred.risk_score)}
                      </span>
                    </td>
                    <td>
                      <span className={`prediction-badge ${pred.prediction === 1 ? 'deterioration' : 'stable'}`}>
                        {pred.prediction === 1 
                          ? '⚠️ تدهور - Deterioration' 
                          : '✅ مستقر - Stable'}
                      </span>
                    </td>
                    <td className="confidence-cell">
                      {(pred.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="date-cell">
                      {new Date(pred.timestamp).toLocaleDateString('ar-PS')}
                      <br/>
                      <small>{new Date(pred.timestamp).toLocaleTimeString('ar-PS')}</small>
                    </td>
                    <td className="no-print">
                      <button 
                        className="btn-view-details"
                        onClick={() => setSelectedPrediction(pred)}
                      >
                        👁️ عرض
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="table-footer no-print">
            <p>عرض {filteredPredictions.length} من {predictions.length} سجل</p>
            <p>Showing {filteredPredictions.length} of {predictions.length} records</p>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {selectedPrediction && (
        <div className="modal-overlay no-print" onClick={() => setSelectedPrediction(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📋 تفاصيل التنبؤ - Prediction Details</h3>
              <button 
                className="modal-close"
                onClick={() => setSelectedPrediction(null)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              {/* Risk Display */}
              <div 
                className="modal-risk-display"
                style={{ 
                  background: `linear-gradient(135deg, ${getRiskColor(selectedPrediction.risk_score)}, ${getRiskColor(selectedPrediction.risk_score)}dd)` 
                }}
              >
                <div className="modal-risk-category">
                  {getRiskCategory(selectedPrediction.risk_score)}
                </div>
                <div className="modal-risk-score">
                  {(selectedPrediction.risk_score * 100).toFixed(1)}%
                </div>
                <p>درجة الخطر - Risk Score</p>
              </div>

              {/* Details Grid */}
              <div className="modal-details-grid">
                <div className="modal-detail-item">
                  <span className="detail-label">رقم المريض - Patient ID:</span>
                  <span className="detail-value">{selectedPrediction.patient_id}</span>
                </div>
                <div className="modal-detail-item">
                  <span className="detail-label">التنبؤ - Prediction:</span>
                  <span className="detail-value">
                    {selectedPrediction.prediction === 1 
                      ? 'تدهور متوقع - Deterioration' 
                      : 'مستقر - Stable'}
                  </span>
                </div>
                <div className="modal-detail-item">
                  <span className="detail-label">الثقة - Confidence:</span>
                  <span className="detail-value">
                    {(selectedPrediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="modal-detail-item">
                  <span className="detail-label">التاريخ - Timestamp:</span>
                  <span className="detail-value">
                    {new Date(selectedPrediction.timestamp).toLocaleString('ar-PS')}
                  </span>
                </div>
              </div>

              {/* Patient Data */}
              {selectedPrediction.input_data && (
                <div className="modal-section">
                  <h4>📊 بيانات المريض - Patient Data</h4>
                  <div className="patient-data-grid">
                    <div className="data-item">
                      <span>العمر - Age:</span>
                      <strong>{selectedPrediction.input_data.age}</strong>
                    </div>
                    <div className="data-item">
                      <span>الجنس - Gender:</span>
                      <strong>{selectedPrediction.input_data.gender === 1 ? 'ذكر - Male' : 'أنثى - Female'}</strong>
                    </div>
                    <div className="data-item">
                      <span>نبض القلب - HR:</span>
                      <strong>{selectedPrediction.input_data.heart_rate}</strong>
                    </div>
                    <div className="data-item">
                      <span>SpO2:</span>
                      <strong>{selectedPrediction.input_data.spo2_pct}%</strong>
                    </div>
                    {selectedPrediction.input_data.temperature && (
                      <div className="data-item">
                        <span>Temp:</span>
                        <strong>{selectedPrediction.input_data.temperature}°C</strong>
                      </div>
                    )}
                    {selectedPrediction.input_data.lactate && (
                      <div className="data-item">
                        <span>Lactate:</span>
                        <strong>{selectedPrediction.input_data.lactate}</strong>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Analysis Summary (if from CSV upload) */}
              {selectedPrediction.analysis_summary && (
                <div className="modal-section">
                  <h4>📈 ملخص التحليل - Analysis Summary</h4>
                  <div className="patient-data-grid">
                    <div className="data-item">
                      <span>أدنى خطر - Min Risk:</span>
                      <strong>{(selectedPrediction.analysis_summary.min_risk * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="data-item">
                      <span>متوسط - Avg Risk:</span>
                      <strong>{(selectedPrediction.analysis_summary.avg_risk * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="data-item">
                      <span>أعلى خطر - Max Risk:</span>
                      <strong>{(selectedPrediction.analysis_summary.max_risk * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="data-item">
                      <span>الاتجاه - Trend:</span>
                      <strong>{selectedPrediction.analysis_summary.risk_trend}</strong>
                    </div>
                    <div className="data-item">
                      <span>ساعات خطر عالي:</span>
                      <strong>{selectedPrediction.analysis_summary.hours_high_risk}</strong>
                    </div>
                    <div className="data-item">
                      <span>ساعات خطر متوسط:</span>
                      <strong>{selectedPrediction.analysis_summary.hours_medium_risk}</strong>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {selectedPrediction.recommendations && (
                <div className="modal-section">
                  <h4>💡 التوصيات - Recommendations</h4>
                  <ul className="modal-recommendations">
                    {selectedPrediction.recommendations.map((rec, idx) => (
                      <li key={idx}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Modal Actions */}
              <div className="modal-actions">
                <button 
                  className="btn-primary"
                  onClick={() => printPrediction(selectedPrediction)}
                >
                  🖨️ طباعة - Print
                </button>
                <button 
                  className="btn-secondary"
                  onClick={() => setSelectedPrediction(null)}
                >
                  ✕ إغلاق - Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;