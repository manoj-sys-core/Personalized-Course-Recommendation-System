import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, CheckCircle, Clock, Play, RotateCcw, AlertCircle } from 'lucide-react';
import { toast } from 'react-toastify';

import { useAppContext } from '../context/AppContext';
import { apiService } from '../services/api';
import { actions } from '../context/AppContext';
import LoadingSpinner from './LoadingSpinner';

const ModelTraining = () => {
  const navigate = useNavigate();
  const { state, dispatch } = useAppContext();
  const [activeTraining, setActiveTraining] = useState({});

  useEffect(() => {
    // Redirect if no user
    if (!state.user.id) {
      navigate('/');
      return;
    }

    // Load user models if not already loaded
    if (state.user.trainedModels.length === 0 && state.user.untrainedModels.length === 0) {
      loadUserModels();
    }
  }, [state.user.id, navigate]);

  const loadUserModels = async () => {
    try {
      const response = await apiService.getUserModels(state.user.id);
      dispatch(actions.setUserModels(response.data.data));
    } catch (error) {
      console.error('Error loading user models:', error);
      toast.error('Failed to load model information');
    }
  };

  const trainModel = async (modelName) => {
    try {
      setActiveTraining(prev => ({ ...prev, [modelName]: 'starting' }));
      
      const response = await apiService.trainModel(state.user.id, modelName);
      const taskId = response.data.task_id;
      
      // Poll for training status
      pollTrainingStatus(taskId, modelName);
      
      toast.info(`Training started for ${modelName}`);
    } catch (error) {
      console.error('Error starting training:', error);
      toast.error(`Failed to start training for ${modelName}`);
      setActiveTraining(prev => ({ ...prev, [modelName]: null }));
    }
  };

  const pollTrainingStatus = async (taskId, modelName) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await apiService.getTrainingStatus(taskId);
        const status = response.data.data;
        
        setActiveTraining(prev => ({ ...prev, [modelName]: status.status }));
        
        if (status.status === 'completed') {
          clearInterval(pollInterval);
          toast.success(`${modelName} training completed!`);
          
          // Reload user models
          await loadUserModels();
          setActiveTraining(prev => ({ ...prev, [modelName]: null }));
          
        } else if (status.status === 'failed') {
          clearInterval(pollInterval);
          toast.error(`${modelName} training failed: ${status.result}`);
          setActiveTraining(prev => ({ ...prev, [modelName]: null }));
        }
      } catch (error) {
        console.error('Error polling training status:', error);
        clearInterval(pollInterval);
        setActiveTraining(prev => ({ ...prev, [modelName]: null }));
      }
    }, 2000);

    // Cleanup after 10 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      setActiveTraining(prev => ({ ...prev, [modelName]: null }));
    }, 600000);
  };

  const modelDescriptions = {
    'Course Similarity': 'Content-based filtering using course descriptions and features to find similar courses',
    'User Profile': 'Creates user profiles based on course genres and preferences for personalized recommendations',
    'Clustering': 'Groups similar users using K-Means clustering to recommend popular courses within clusters',
    'Clustering with PCA': 'Advanced clustering with Principal Component Analysis for improved user grouping',
    'Neural Network': 'Deep Neural Collaborative Filtering for sophisticated recommendation patterns',
    'Regression with Embedding Features': 'Uses learned user and item embeddings with regression for rating prediction',
    'Classification with Embedding Features': 'Classification approach with embeddings for binary recommendation decisions'
  };

  const modelComplexity = {
    'Course Similarity': 'Low',
    'User Profile': 'Low', 
    'Clustering': 'Medium',
    'Clustering with PCA': 'Medium',
    'Neural Network': 'High',
    'Regression with Embedding Features': 'High',
    'Classification with Embedding Features': 'High'
  };

  const getComplexityColor = (complexity) => {
    switch (complexity) {
      case 'Low': return 'text-green-600 bg-green-100';
      case 'Medium': return 'text-yellow-600 bg-yellow-100';
      case 'High': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const ModelCard = ({ modelName, isTrained, showTrainButton = true }) => {
    const trainingStatus = activeTraining[modelName];
    const isTraining = trainingStatus === 'training' || trainingStatus === 'starting';
    
    return (
      <div className={`card ${isTraining ? 'ring-2 ring-primary-500' : ''}`}>
        <div className="card-body">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                isTrained ? 'bg-green-100' : 'bg-gray-100'
              }`}>
                {isTraining ? (
                  <LoadingSpinner size="small" />
                ) : isTrained ? (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                ) : (
                  <Brain className="w-6 h-6 text-gray-600" />
                )}
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">{modelName}</h3>
                <div className="flex items-center space-x-2 mt-1">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    getComplexityColor(modelComplexity[modelName])
                  }`}>
                    {modelComplexity[modelName]} Complexity
                  </span>
                  {isTrained && (
                    <span className="px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                      Trained
                    </span>
                  )}
                </div>
              </div>
            </div>
            
            {showTrainButton && !isTrained && (
              <button
                onClick={() => trainModel(modelName)}
                disabled={isTraining}
                className="btn-primary"
              >
                {isTraining ? (
                  <>
                    <Clock className="w-4 h-4 mr-2" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Train
                  </>
                )}
              </button>
            )}
          </div>
          
          <p className="text-sm text-gray-600 mb-4">
            {modelDescriptions[modelName]}
          </p>
          
          {isTraining && (
            <div className="bg-primary-50 border border-primary-200 rounded-lg p-3">
              <div className="flex items-center space-x-2">
                <LoadingSpinner size="small" />
                <span className="text-sm text-primary-700 font-medium">
                  Training in progress... This may take several minutes.
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (!state.user.id) {
    return (
      <div className="text-center py-16">
        <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">No User Selected</h3>
        <p className="text-gray-600 mb-4">Please select a user or create a new account to train models.</p>
        <button 
          onClick={() => navigate('/')}
          className="btn-primary"
        >
          Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Model Training</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Train different machine learning models to get personalized course recommendations. 
          Each model uses different algorithms and approaches.
        </p>
      </div>

      {/* User Info */}
      <div className="card">
        <div className="card-body">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-gray-900">Training for User {state.user.id}</h3>
              <p className="text-sm text-gray-600 mt-1">
                {state.user.completedCourses.length} completed courses • 
                {state.user.trainedModels.length} trained models • 
                {state.user.untrainedModels.length} untrained models
              </p>
            </div>
            <button
              onClick={loadUserModels}
              className="btn-secondary"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Trained Models */}
      {state.user.trainedModels.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <CheckCircle className="w-5 h-5 text-green-600 mr-2" />
            Trained Models ({state.user.trainedModels.length})
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {state.user.trainedModels.map(modelName => (
              <ModelCard
                key={modelName}
                modelName={modelName}
                isTrained={true}
                showTrainButton={false}
              />
            ))}
          </div>
        </div>
      )}

      {/* Untrained Models */}
      {state.user.untrainedModels.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Brain className="w-5 h-5 text-gray-600 mr-2" />
            Available for Training ({state.user.untrainedModels.length})
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {state.user.untrainedModels.map(modelName => (
              <ModelCard
                key={modelName}
                modelName={modelName}
                isTrained={false}
                showTrainButton={true}
              />
            ))}
          </div>
        </div>
      )}

      {/* Training Tips */}
      <div className="card bg-blue-50 border-blue-200">
        <div className="card-body">
          <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            Training Tips
          </h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li>• Models with higher complexity may take longer to train but provide better accuracy</li>
            <li>• You can train multiple models and compare their recommendations</li>
            <li>• Neural Network and Embedding models require more computational resources</li>
            <li>• Adding more completed courses will improve model accuracy</li>
          </ul>
        </div>
      </div>

      {/* Action Buttons */}
      {state.user.trainedModels.length > 0 && (
        <div className="text-center">
          <button 
            onClick={() => navigate('/recommendations')}
            className="btn-primary text-lg px-8 py-3"
          >
            View Recommendations
          </button>
        </div>
      )}
    </div>
  );
};

export default ModelTraining;