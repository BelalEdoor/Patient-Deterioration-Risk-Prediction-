import React, { useState, useEffect } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import PredictPatient from './pages/PredictPatient';
import PatientHistory from './pages/PatientHistory';
import './App.css';

function App() {
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkModelStatus();
    const interval = setInterval(checkModelStatus, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/model/status');
      const data = await response.json();
      setModelStatus(data);
    } catch (error) {
      console.error('Error checking model status:', error);
      setModelStatus({ trained: false, error: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <nav className="navbar">
        <div className="navbar-container">
          <div className="navbar-brand">
            <h1>🏥 نظام التنبؤ بخطر المرضى</h1>
            <p className="brand-subtitle">Patient Deterioration Risk Prediction</p>
          </div>

          <ul className="navbar-menu">
            <li><Link to="/">📊 Dashboard</Link></li>
            <li><Link to="/predict">🔮 Predict</Link></li>
            <li><Link to="/history">📚 History</Link></li>
          </ul>

          <div className="model-status">
            {loading ? (
              <span className="status-badge status-loading">⏳ Loading...</span>
            ) : modelStatus?.trained ? (
              <span className="status-badge status-trained">✅ Model Ready</span>
            ) : (
              <span className="status-badge status-not-trained">⚠️ Not Trained</span>
            )}
          </div>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route
            path="/"
            element={<Dashboard modelStatus={modelStatus} />}
          />
          <Route
            path="/predict"
            element={<PredictPatient modelStatus={modelStatus} />}
          />
          <Route
            path="/history"
            element={<PatientHistory />}
          />
        </Routes>
      </main>

      <footer className="footer">
        <div className="footer-container">
          <div className="footer-section">
            <h3>🏥 Hospital Patient Risk System</h3>
            <p>Advanced AI-powered patient deterioration prediction</p>
          </div>
          <div className="footer-section">
            <h4>Developers</h4>
            <p>Belal Edoor • Mohammad Alhoor</p>
          </div>
          <div className="footer-section">
            <h4>Technology</h4>
            <p>React • FastAPI • Random Forest ML</p>
          </div>
        </div>
        <div className="footer-bottom">
          <p>© 2026 Hospital Patient Deterioration Prediction System | Made with ❤️ for Healthcare</p>
        </div>
      </footer>
    </div>
  );
}

export default App;