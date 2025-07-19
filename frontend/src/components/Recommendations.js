import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Star, BookOpen, TrendingUp, RefreshCw, AlertCircle, Target } from 'lucide-react';
import { toast } from 'react-toastify';

import { useAppContext } from '../context/AppContext';
import { apiService } from '../services/api';
import { actions } from '../context/AppContext';
import LoadingSpinner from './LoadingSpinner';

const Recommendations = () => {
  const navigate = useNavigate();
  const { state, dispatch } = useAppContext();
  const [selectedModel, setSelectedModel] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [nRecommendations, setNRecommendations] = useState(10);

  useEffect(() => {
    // Redirect if no user
    if (!state.user.id) {
      navigate('/');
      return;
    }

    // Load user models if not already loaded
    if (state.user.trainedModels.length === 0) {
      loadUserModels();
    } else if (state.user.trainedModels.length > 0 && !selectedModel) {
      setSelectedModel(state.user.trainedModels[0]);
    }
  }, [state.user.id, navigate, state.user.trainedModels]);

  const loadUserModels = async () => {
    try {
      const response = await apiService.getUserModels(state.user.id);
      dispatch(actions.setUserModels(response.data.data));
      
      // Set first trained model as default
      if (response.data.data.trained_models && response.data.data.trained_models.length > 0) {
        setSelectedModel(response.data.data.trained_models[0]);
      }
    } catch (error) {
      console.error('Error loading user models:', error);
      toast.error('Failed to load model information');
    }
  };

  const generateRecommendations = async () => {
    if (!selectedModel) {
      toast.warning('Please select a model first');
      return;
    }

    try {
      setLoading(true);
      const response = await apiService.getPredictions(
        state.user.id,
        selectedModel,
        nRecommendations
      );
      
      setRecommendations(response.data.data);
      dispatch(actions.setRecommendations(response.data.data));
      
      toast.success(`Generated ${response.data.data.length} recommendations using ${selectedModel}`);
    } catch (error) {
      console.error('Error generating recommendations:', error);
      toast.error('Failed to generate recommendations');
    } finally {
      setLoading(false);
    }
  };

  // Auto-generate recommendations when model is selected
  useEffect(() => {
    if (selectedModel && state.user.id) {
      generateRecommendations();
    }
  }, [selectedModel]);

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    if (score >= 40) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  const getScoreLabel = (score) => {
    if (score >= 80) return 'Excellent Match';
    if (score >= 60) return 'Good Match';
    if (score >= 40) return 'Fair Match';
    return 'Consider';
  };

  const RecommendationCard = ({ recommendation, index }) => (
    <div className="card hover:shadow-lg transition-all duration-200">
      <div className="card-body">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-100 rounded-xl flex items-center justify-center">
              <span className="font-bold text-primary-600">#{index + 1}</span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 leading-tight">
                {recommendation.TITLE}
              </h3>
              <p className="text-sm text-gray-500">Course ID: {recommendation.COURSE_ID}</p>
            </div>
          </div>
          
          <div className="flex flex-col items-end space-y-2">
            <span className={`px-3 py-1 rounded-full text-sm font-bold ${
              getScoreColor(recommendation.SCORE)
            }`}>
              {recommendation.SCORE}%
            </span>
            <span className="text-xs text-gray-500">
              {getScoreLabel(recommendation.SCORE)}
            </span>
          </div>
        </div>
        
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <div className="flex items-center space-x-1">
            <Target className="w-4 h-4" />
            <span>Confidence: {recommendation.SCORE}%</span>
          </div>
          <div className="flex items-center space-x-1">
            <Star className="w-4 h-4" />
            <span>Rank: {index + 1}</span>
          </div>
        </div>
      </div>
    </div>
  );

  if (!state.user.id) {
    return (
      <div className="text-center py-16">
        <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No User Selected</h3>
        <p className="text-gray-600 mb-4">Please select a user to view recommendations.</p>
        <button 
          onClick={() => navigate('/')}
          className="btn-primary"
        >
          Go to Dashboard
        </button>
      </div>
    );
  }

  if (state.user.trainedModels.length === 0) {
    return (
      <div className="text-center py-16">
        <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No Trained Models</h3>
        <p className="text-gray-600 mb-4">
          You need to train at least one model before getting recommendations.
        </p>
        <button 
          onClick={() => navigate('/training')}
          className="btn-primary"
        >
          Train Models
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Course Recommendations</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Get personalized course recommendations based on your completed courses and trained models.
        </p>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="input-field"
              >
                <option value="">Choose a model...</option>
                {state.user.trainedModels.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Recommendations
              </label>
              <select
                value={nRecommendations}
                onChange={(e) => setNRecommendations(parseInt(e.target.value))}
                className="input-field"
              >
                <option value={5}>5 recommendations</option>
                <option value={10}>10 recommendations</option>
                <option value={15}>15 recommendations</option>
                <option value={20}>20 recommendations</option>
              </select>
            </div>
            
            <button
              onClick={generateRecommendations}
              disabled={!selectedModel || loading}
              className="btn-primary"
            >
              {loading ? (
                <>
                  <LoadingSpinner size="small" />
                  <span className="ml-2">Generating...</span>
                </>
              ) : (
                <>
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Generate
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* User Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <div className="card-body text-center">
            <BookOpen className="w-8 h-8 text-primary-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {state.user.completedCourses.length}
            </div>
            <div className="text-sm text-gray-600">Completed Courses</div>
          </div>
        </div>
        
        <div className="card">
          <div className="card-body text-center">
            <Star className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {recommendations.length}
            </div>
            <div className="text-sm text-gray-600">Recommendations</div>
          </div>
        </div>
        
        <div className="card">
          <div className="card-body text-center">
            <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">
              {recommendations.length > 0 ? Math.round(recommendations[0]?.SCORE || 0) : 0}%
            </div>
            <div className="text-sm text-gray-600">Top Match Score</div>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 ? (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              Recommendations from {selectedModel}
            </h2>
            <span className="text-sm text-gray-600">
              Showing {recommendations.length} results
            </span>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {recommendations.map((recommendation, index) => (
              <RecommendationCard
                key={recommendation.COURSE_ID}
                recommendation={recommendation}
                index={index}
              />
            ))}
          </div>
        </div>
      ) : !loading && selectedModel && (
        <div className="text-center py-16">
          <Star className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Recommendations</h3>
          <p className="text-gray-600">
            No recommendations available for the selected model. Try training more models or adding more courses.
          </p>
        </div>
      )}

      {/* Model Performance Info */}
      {selectedModel && recommendations.length > 0 && (
        <div className="card bg-gray-50">
          <div className="card-body">
            <h3 className="font-semibold text-gray-900 mb-3">Model Performance</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Average Score:</span>
                <div className="font-semibold text-gray-900">
                  {Math.round(recommendations.reduce((acc, rec) => acc + rec.SCORE, 0) / recommendations.length)}%
                </div>
              </div>
              <div>
                <span className="text-gray-600">Highest Score:</span>
                <div className="font-semibold text-gray-900">
                  {Math.max(...recommendations.map(rec => rec.SCORE))}%
                </div>
              </div>
              <div>
                <span className="text-gray-600">Lowest Score:</span>
                <div className="font-semibold text-gray-900">
                  {Math.min(...recommendations.map(rec => rec.SCORE))}%
                </div>
              </div>
              <div>
                <span className="text-gray-600">Model Type:</span>
                <div className="font-semibold text-gray-900">{selectedModel}</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Recommendations;