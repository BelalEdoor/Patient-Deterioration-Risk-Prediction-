import React, { useState } from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import PredictPatient from './pages/PredictPatient';
import PatientManagement from './pages/Patientmanagement';
import PatientDetails from './pages/Patientdetails';
import './App.css';

function App() {
  const [modelStatus] = useState({ trained: true });

  return (
    <div className="App">
      <nav className="main-nav">
        <div className="nav-container">
          <Link to="/" className="nav-logo">
            <span className="logo-icon">🏥</span>
            <span className="logo-text">Patient Risk System</span>
          </Link>

          <div className="nav-links">
            <Link to="/" className="nav-link">📊 Dashboard</Link>
            <Link to="/patients" className="nav-link">👥 Patients</Link>
            <Link to="/predict" className="nav-link">📈 CSV Analysis</Link>
          </div>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patients" element={<PatientManagement />} />
          <Route path="/patient/:patientId" element={<PatientDetails />} />
          <Route path="/predict" element={<PredictPatient modelStatus={modelStatus} />} />
        </Routes>
      </main>

      <footer className="main-footer">
        <p>© 2026 Hospital Patient Risk System</p>
        <p>نظام التنبؤ بتدهور حالة المرضى</p>
      </footer>
    </div>
  );
}

export default App;
