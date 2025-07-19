import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { GraduationCap, User, Brain, Star } from 'lucide-react';
import { useAppContext } from '../context/AppContext';

const Header = () => {
  const location = useLocation();
  const { state } = useAppContext();

  const navigationItems = [
    { path: '/', label: 'Dashboard', icon: GraduationCap },
    { path: '/courses', label: 'Courses', icon: User },
    { path: '/training', label: 'Training', icon: Brain },
    { path: '/recommendations', label: 'Recommendations', icon: Star },
  ];

  return (
    <header className="gradient-bg text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
              <GraduationCap className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Course Recommender</h1>
              <p className="text-xs text-white/80">AI-Powered Learning</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-1">
            {navigationItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-white/20 text-white shadow-sm'
                      : 'text-white/80 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>

          {/* User Info */}
          <div className="flex items-center space-x-3">
            {state.user.id && (
              <div className="hidden sm:flex items-center space-x-2 bg-white/10 rounded-lg px-3 py-1">
                <User className="w-4 h-4" />
                <span className="text-sm font-medium">User {state.user.id}</span>
              </div>
            )}
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden pb-4">
          <nav className="flex space-x-1 overflow-x-auto">
            {navigationItems.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link
                  key={path}
                  to={path}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-all duration-200 ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : 'text-white/80 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;