import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Components
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import CourseSelection from './components/CourseSelection';
import ModelTraining from './components/ModelTraining';
import Recommendations from './components/Recommendations';
import LoadingSpinner from './components/LoadingSpinner';

// Services
import api from './services/api';

// Context
import { AppProvider } from './context/AppContext';

function App() {
  const [loading, setLoading] = useState(true);
  const [apiHealthy, setApiHealthy] = useState(false);

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await api.get('/api/health');
      if (response.data.status === 'healthy') {
        setApiHealthy(true);
      }
    } catch (error) {
      console.error('API health check failed:', error);
      setApiHealthy(false);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (!apiHealthy) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="mb-4">
            <div className="mx-auto w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
              <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">API Connection Failed</h3>
          <p className="text-gray-500 mb-4">Unable to connect to the backend API. Please ensure the server is running.</p>
          <button 
            onClick={checkApiHealth}
            className="btn-primary"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <AppProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Header />
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/courses" element={<CourseSelection />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/recommendations" element={<Recommendations />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
          <ToastContainer
            position="top-right"
            autoClose={4000}
            hideProgressBar={false}
            newestOnTop={false}
            closeOnClick
            rtl={false}
            pauseOnFocusLoss
            draggable
            pauseOnHover
            theme="light"
          />
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;