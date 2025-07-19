import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.response) {
      // Server responded with error status
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // Request was made but no response
      console.error('Network Error:', error.request);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// API service functions
export const apiService = {
  // Health check
  checkHealth: () => api.get('/api/health'),

  // Courses
  getCourses: () => api.get('/api/courses'),
  getUserCourses: (userId) => api.get(`/api/user/${userId}/courses`),

  // Users
  getAllUsers: () => api.get('/api/users'),
  createNewUser: (courseIds) => api.post('/api/user/new', { course_ids: courseIds }),
  addUserCourses: (userId, courseIds) => api.post(`/api/user/${userId}/courses/add`, { course_ids: courseIds }),

  // Models
  getUserModels: (userId) => api.get(`/api/user/${userId}/models`),
  trainModel: (userId, modelName) => api.post(`/api/user/${userId}/train`, { 
    user_id: userId, 
    model_name: modelName 
  }),
  getTrainingStatus: (taskId) => api.get(`/api/training/${taskId}/status`),

  // Recommendations
  getPredictions: (userId, modelName, nRecommendations = 10) => api.post(`/api/user/${userId}/predict`, {
    user_id: userId,
    model_name: modelName,
    n_recommendations: nRecommendations
  }),

  // Ratings
  getRatings: () => api.get('/api/ratings'),
};

export default api;