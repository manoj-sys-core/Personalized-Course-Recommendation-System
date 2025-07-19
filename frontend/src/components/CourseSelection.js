import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, BookOpen, Plus, Check, X } from 'lucide-react';
import { toast } from 'react-toastify';

import { useAppContext } from '../context/AppContext';
import { apiService } from '../services/api';
import { actions } from '../context/AppContext';
import LoadingSpinner from './LoadingSpinner';

const CourseSelection = () => {
  const navigate = useNavigate();
  const { state, dispatch } = useAppContext();
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (state.courses.length === 0) {
      loadCourses();
    }
    
    // If user is not initialized, redirect to dashboard
    if (!state.user.id && state.user.isNewUser === null) {
      navigate('/');
    }
  }, [state.user, navigate, state.courses.length]);

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

  const getAvailableCourses = () => {
    const completedCourseIds = state.user.completedCourses.map(course => course.COURSE_ID);
    return state.courses.filter(course => !completedCourseIds.includes(course.COURSE_ID));
  };

  const filteredCourses = getAvailableCourses().filter(course =>
    course.TITLE.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.DESCRIPTION.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const toggleCourseSelection = (courseId) => {
    setSelectedCourses(prev => 
      prev.includes(courseId)
        ? prev.filter(id => id !== courseId)
        : [...prev, courseId]
    );
  };

  const handleSubmit = async () => {
    if (selectedCourses.length === 0) {
      toast.warning('Please select at least one course');
      return;
    }

    try {
      setSubmitting(true);
      
      if (state.user.isNewUser) {
        // Create new user
        const response = await apiService.createNewUser(selectedCourses);
        const newUserId = response.data.data.user_id;
        
        dispatch(actions.setUser({ 
          id: newUserId, 
          isNewUser: false 
        }));
        
        // Load user courses and models
        const [coursesResponse, modelsResponse] = await Promise.all([
          apiService.getUserCourses(newUserId),
          apiService.getUserModels(newUserId),
        ]);
        
        dispatch(actions.setUserCourses(coursesResponse.data.data));
        dispatch(actions.setUserModels(modelsResponse.data.data));
        
        toast.success(`Welcome! Your user ID is ${newUserId}`);
        navigate('/training');
      } else {
        // Add courses to existing user
        await apiService.addUserCourses(state.user.id, selectedCourses);
        
        // Reload user data
        const [coursesResponse, modelsResponse] = await Promise.all([
          apiService.getUserCourses(state.user.id),
          apiService.getUserModels(state.user.id),
        ]);
        
        dispatch(actions.setUserCourses(coursesResponse.data.data));
        dispatch(actions.setUserModels(modelsResponse.data.data));
        
        toast.success('Courses added successfully!');
        navigate('/');
      }
    } catch (error) {
      console.error('Error submitting courses:', error);
      toast.error('Failed to submit courses');
    } finally {
      setSubmitting(false);
    }
  };

  const CourseCard = ({ course }) => {
    const isSelected = selectedCourses.includes(course.COURSE_ID);
    
    return (
      <div 
        className={`card cursor-pointer transition-all duration-200 ${
          isSelected 
            ? 'ring-2 ring-primary-500 bg-primary-50' 
            : 'hover:shadow-md'
        }`}
        onClick={() => toggleCourseSelection(course.COURSE_ID)}
      >
        <div className="card-body">
          <div className="flex items-start justify-between">
            <div className="flex-1 pr-4">
              <div className="flex items-center space-x-3 mb-2">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                  isSelected ? 'bg-primary-600 text-white' : 'bg-gray-100'
                }`}>
                  {isSelected ? (
                    <Check className="w-5 h-5" />
                  ) : (
                    <BookOpen className="w-5 h-5 text-gray-600" />
                  )}
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 leading-tight">
                    {course.TITLE}
                  </h3>
                  <p className="text-sm text-gray-500">ID: {course.COURSE_ID}</p>
                </div>
              </div>
              <p className="text-sm text-gray-600 line-clamp-2">
                {course.DESCRIPTION}
              </p>
            </div>
            <div className={`w-6 h-6 rounded-full border-2 flex items-center justify-center ${
              isSelected 
                ? 'bg-primary-600 border-primary-600' 
                : 'border-gray-300'
            }`}>
              {isSelected && <Check className="w-4 h-4 text-white" />}
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (state.loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <LoadingSpinner size="large" text="Loading courses..." />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          {state.user.isNewUser ? 'Select Your Completed Courses' : 'Add More Courses'}
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          {state.user.isNewUser 
            ? 'Choose courses you have already completed to help us understand your learning preferences and provide better recommendations.'
            : 'Add additional courses you have completed to improve your recommendations.'
          }
        </p>
      </div>

      {/* Current User Info */}
      {!state.user.isNewUser && (
        <div className="card">
          <div className="card-body">
            <h3 className="font-semibold text-gray-900 mb-2">Current Completed Courses</h3>
            <div className="flex flex-wrap gap-2">
              {state.user.completedCourses.map(course => (
                <span 
                  key={course.COURSE_ID}
                  className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-green-100 text-green-800"
                >
                  <Check className="w-3 h-3 mr-1" />
                  {course.TITLE}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search and Selection Stats */}
      <div className="flex flex-col sm:flex-row gap-4 items-center justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search courses..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field pl-10"
          />
        </div>
        
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-600">
            {selectedCourses.length} selected • {filteredCourses.length} available
          </span>
          
          {selectedCourses.length > 0 && (
            <button
              onClick={() => setSelectedCourses([])}
              className="text-sm text-red-600 hover:text-red-700 flex items-center"
            >
              <X className="w-4 h-4 mr-1" />
              Clear All
            </button>
          )}
        </div>
      </div>

      {/* Course Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {filteredCourses.map(course => (
          <CourseCard key={course.COURSE_ID} course={course} />
        ))}
      </div>

      {/* No Results */}
      {filteredCourses.length === 0 && (
        <div className="text-center py-16">
          <BookOpen className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No courses found</h3>
          <p className="text-gray-600">
            {searchTerm ? 'Try adjusting your search terms' : 'All available courses have been completed'}
          </p>
        </div>
      )}

      {/* Submit Button */}
      {selectedCourses.length > 0 && (
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-white rounded-full shadow-lg border border-gray-200 px-6 py-3">
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-900">
                {selectedCourses.length} course{selectedCourses.length !== 1 ? 's' : ''} selected
              </span>
              <button
                onClick={handleSubmit}
                disabled={submitting}
                className="btn-primary"
              >
                {submitting ? (
                  <>
                    <LoadingSpinner size="small" />
                    <span className="ml-2">
                      {state.user.isNewUser ? 'Creating Account...' : 'Adding Courses...'}
                    </span>
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4 mr-2" />
                    {state.user.isNewUser ? 'Create Account' : 'Add Courses'}
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CourseSelection;