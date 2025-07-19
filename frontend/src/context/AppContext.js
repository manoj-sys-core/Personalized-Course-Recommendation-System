import React, { createContext, useContext, useReducer } from 'react';

// Initial state
const initialState = {
  user: {
    id: null,
    isNewUser: null,
    completedCourses: [],
    trainedModels: [],
    untrainedModels: [],
  },
  courses: [],
  recommendations: [],
  loading: false,
  error: null,
  trainingStatus: {},
};

// Action types
export const ACTION_TYPES = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  CLEAR_ERROR: 'CLEAR_ERROR',
  SET_COURSES: 'SET_COURSES',
  SET_USER: 'SET_USER',
  SET_USER_COURSES: 'SET_USER_COURSES',
  SET_USER_MODELS: 'SET_USER_MODELS',
  SET_RECOMMENDATIONS: 'SET_RECOMMENDATIONS',
  UPDATE_TRAINING_STATUS: 'UPDATE_TRAINING_STATUS',
  RESET_USER: 'RESET_USER',
};

// Reducer
const appReducer = (state, action) => {
  switch (action.type) {
    case ACTION_TYPES.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
      };

    case ACTION_TYPES.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false,
      };

    case ACTION_TYPES.CLEAR_ERROR:
      return {
        ...state,
        error: null,
      };

    case ACTION_TYPES.SET_COURSES:
      return {
        ...state,
        courses: action.payload,
      };

    case ACTION_TYPES.SET_USER:
      return {
        ...state,
        user: {
          ...state.user,
          ...action.payload,
        },
      };

    case ACTION_TYPES.SET_USER_COURSES:
      return {
        ...state,
        user: {
          ...state.user,
          completedCourses: action.payload,
        },
      };

    case ACTION_TYPES.SET_USER_MODELS:
      return {
        ...state,
        user: {
          ...state.user,
          trainedModels: action.payload.trained_models || [],
          untrainedModels: action.payload.untrained_models || [],
        },
      };

    case ACTION_TYPES.SET_RECOMMENDATIONS:
      return {
        ...state,
        recommendations: action.payload,
      };

    case ACTION_TYPES.UPDATE_TRAINING_STATUS:
      return {
        ...state,
        trainingStatus: {
          ...state.trainingStatus,
          [action.payload.taskId]: action.payload.status,
        },
      };

    case ACTION_TYPES.RESET_USER:
      return {
        ...state,
        user: initialState.user,
        recommendations: [],
        trainingStatus: {},
      };

    default:
      return state;
  }
};

// Context
const AppContext = createContext();

// Provider component
export const AppProvider = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const value = {
    state,
    dispatch,
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};

// Hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
};

// Action creators
export const actions = {
  setLoading: (loading) => ({
    type: ACTION_TYPES.SET_LOADING,
    payload: loading,
  }),

  setError: (error) => ({
    type: ACTION_TYPES.SET_ERROR,
    payload: error,
  }),

  clearError: () => ({
    type: ACTION_TYPES.CLEAR_ERROR,
  }),

  setCourses: (courses) => ({
    type: ACTION_TYPES.SET_COURSES,
    payload: courses,
  }),

  setUser: (user) => ({
    type: ACTION_TYPES.SET_USER,
    payload: user,
  }),

  setUserCourses: (courses) => ({
    type: ACTION_TYPES.SET_USER_COURSES,
    payload: courses,
  }),

  setUserModels: (models) => ({
    type: ACTION_TYPES.SET_USER_MODELS,
    payload: models,
  }),

  setRecommendations: (recommendations) => ({
    type: ACTION_TYPES.SET_RECOMMENDATIONS,
    payload: recommendations,
  }),

  updateTrainingStatus: (taskId, status) => ({
    type: ACTION_TYPES.UPDATE_TRAINING_STATUS,
    payload: { taskId, status },
  }),

  resetUser: () => ({
    type: ACTION_TYPES.RESET_USER,
  }),
};