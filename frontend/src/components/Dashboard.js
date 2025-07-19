import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Plus, Users, BookOpen, Brain, Star, TrendingUp } from 'lucide-react';
import { toast } from 'react-toastify';

import { useAppContext } from '../context/AppContext';
import { apiService } from '../services/api';
import { actions } from '../context/AppContext';
import LoadingSpinner from './LoadingSpinner';

const Dashboard = () => {
  const { state, dispatch } = useAppContext();
  const [stats, setStats] = useState({
    totalCourses: 0,
    totalUsers: 0,
  });
  const [loadingStats, setLoadingStats] = useState(true);

  useEffect(() => {
    loadStats();
    if (state.courses.length === 0) {
      loadCourses();
    }
  }, []);

  const loadStats = async () => {
    try {
      const [coursesResponse, usersResponse] = await Promise.all([
        apiService.getCourses(),
        apiService.getAllUsers(),
      ]);

      setStats({
        totalCourses: coursesResponse.data.data.length,
        totalUsers: usersResponse.data.data.length,
      });
    } catch (error) {
      console.error('Error loading stats:', error);
      toast.error('Failed to load statistics');
    } finally {
      setLoadingStats(false);
    }
  };

  const loadCourses = async () => {
    try {
      dispatch(actions.setLoading(true));
      const response = await apiService.getCourses();
      dispatch(actions.setCourses(response.data.data));
    } catch (error) {
      console.error('Error loading courses:', error);
      toast.error('Failed to load courses');
      dispatch(actions.setError('Failed to load courses'));
    } finally {
      dispatch(actions.setLoading(false));
    }
  };

  const handleExistingUser = async (userId) => {
    try {
      dispatch(actions.setLoading(true));
      
      // Load user data
      const [coursesResponse, modelsResponse] = await Promise.all([
        apiService.getUserCourses(userId),
        apiService.getUserModels(userId),
      ]);

      dispatch(actions.setUser({ 
        id: userId, 
        isNewUser: false 
      }));
      dispatch(actions.setUserCourses(coursesResponse.data.data));
      dispatch(actions.setUserModels(modelsResponse.data.data));
      
      toast.success(`Welcome back, User ${userId}!`);
    } catch (error) {
      console.error('Error loading user data:', error);
      toast.error('Failed to load user data');
    } finally {
      dispatch(actions.setLoading(false));
    }
  };

  const startNewUser = () => {
    dispatch(actions.resetUser());
    dispatch(actions.setUser({ isNewUser: true }));
    toast.info('Let\'s get you started! Select your completed courses.');
  };

  const StatCard = ({ icon: Icon, title, value, color = 'primary' }) => (
    <div className="card">
      <div className="card-body">
        <div className="flex items-center">
          <div className={`w-12 h-12 bg-${color}-100 rounded-xl flex items-center justify-center`}>
            <Icon className={`w-6 h-6 text-${color}-600`} />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className="text-2xl font-bold text-gray-900">
              {loadingStats ? (
                <div className="w-8 h-6 bg-gray-200 rounded animate-pulse"></div>
              ) : (
                value
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );

  const FeatureCard = ({ icon: Icon, title, description, color = 'primary' }) => (
    <div className="card hover:shadow-md transition-all duration-200">
      <div className="card-body">
        <div className={`w-12 h-12 bg-${color}-100 rounded-xl flex items-center justify-center mb-4`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
        <p className="text-gray-600 text-sm">{description}</p>
      </div>
    </div>
  );

  const UserCard = ({ userId }) => (
    <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
      <div className="flex items-center space-x-3">
        <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
          <Users className="w-4 h-4 text-primary-600" />
        </div>
        <span className="font-medium text-gray-900">User {userId}</span>
      </div>
      <button
        onClick={() => handleExistingUser(userId)}
        className="px-3 py-1 text-sm text-primary-600 hover:text-primary-700 font-medium"
        disabled={state.loading}
      >
        Select
      </button>
    </div>
  );

  if (state.loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <LoadingSpinner size="large" text="Loading dashboard..." />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center py-12">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to AI Course Recommender
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Discover personalized learning paths with 7 advanced ML algorithms
          </p>
          
          {!state.user.id && (
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={startNewUser}
                className="btn-primary text-lg px-8 py-3"
              >
                <Plus className="w-5 h-5 mr-2" />
                I'm New Here
              </button>
              <div className="text-gray-500">or</div>
              <div className="text-sm text-gray-600">
                Select your existing user ID below
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Current User Status */}
      {state.user.id && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold">Current Session</h2>
          </div>
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                  <Users className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">
                    User {state.user.id}
                  </h3>
                  <p className="text-sm text-gray-600">
                    {state.user.completedCourses.length} completed courses • {state.user.trainedModels.length} trained models
                  </p>
                </div>
              </div>
              <div className="flex space-x-2">
                <Link to="/courses" className="btn-secondary">
                  Manage Courses
                </Link>
                <Link to="/training" className="btn-primary">
                  Train Models
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={BookOpen}
          title="Total Courses"
          value={stats.totalCourses}
          color="primary"
        />
        <StatCard
          icon={Users}
          title="Total Users"
          value={stats.totalUsers}
          color="secondary"
        />
        <StatCard
          icon={Brain}
          title="ML Algorithms"
          value="7"
          color="green"
        />
        <StatCard
          icon={TrendingUp}
          title="Success Rate"
          value="95%"
          color="purple"
        />
      </div>

      {/* Existing Users */}
      {!state.user.id && (
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold">Existing Users</h2>
            <p className="text-sm text-gray-600 mt-1">
              Select your user ID to continue with your learning journey
            </p>
          </div>
          <div className="card-body">
            <div className="max-h-64 overflow-y-auto space-y-2">
              {stats.totalUsers > 0 ? (
                Array.from({ length: Math.min(10, stats.totalUsers) }, (_, i) => (
                  <UserCard key={i + 1} userId={i + 1} />
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>No existing users found. Be the first to get started!</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <FeatureCard
          icon={Brain}
          title="Neural Collaborative Filtering"
          description="Deep learning approach for sophisticated recommendation patterns"
          color="purple"
        />
        <FeatureCard
          icon={TrendingUp}
          title="User Profile Modeling"
          description="Build comprehensive learner profiles based on course preferences"
          color="blue"
        />
        <FeatureCard
          icon={Star}
          title="Course Similarity Analysis"
          description="Find courses similar to your completed ones using content analysis"
          color="green"
        />
      </div>
    </div>
  );
};

export default Dashboard;