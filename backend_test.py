#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Course Recommendation System
Tests all FastAPI endpoints and ML functionality
"""

import requests
import sys
import json
import time
from datetime import datetime

class CourseRecommendationAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_user_id = None
        self.task_id = None

    def log_test(self, name, success, details=""):
        """Log test results"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")

    def make_request(self, method, endpoint, data=None, expected_status=200):
        """Make HTTP request and return response"""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)
            
            success = response.status_code == expected_status
            return success, response
            
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def test_health_check(self):
        """Test API health endpoint"""
        success, response = self.make_request('GET', '/api/health')
        if success:
            data = response.json()
            success = data.get('status') == 'healthy'
        self.log_test("Health Check", success, "" if success else "API not healthy")
        return success

    def test_root_endpoint(self):
        """Test root endpoint"""
        success, response = self.make_request('GET', '/')
        if success:
            data = response.json()
            success = 'message' in data and 'status' in data
        self.log_test("Root Endpoint", success)
        return success

    def test_get_courses(self):
        """Test get all courses"""
        success, response = self.make_request('GET', '/api/courses')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                courses = data['data']
                print(f"   📊 Found {len(courses)} courses")
        self.log_test("Get Courses", success)
        return success

    def test_get_ratings(self):
        """Test get all ratings"""
        success, response = self.make_request('GET', '/api/ratings')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                ratings = data['data']
                print(f"   📊 Found {len(ratings)} ratings")
        self.log_test("Get Ratings", success)
        return success

    def test_get_users(self):
        """Test get all users"""
        success, response = self.make_request('GET', '/api/users')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                users = data['data']
                print(f"   📊 Found {len(users)} users")
                if users:
                    self.test_user_id = users[0]  # Use first user for testing
        self.log_test("Get Users", success)
        return success

    def test_create_new_user(self):
        """Test creating a new user"""
        test_courses = [1, 2, 3, 4, 5]  # Sample course IDs
        success, response = self.make_request('POST', '/api/user/new', 
                                            {'course_ids': test_courses}, 201)
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                self.test_user_id = data['data']['user_id']
                print(f"   👤 Created user ID: {self.test_user_id}")
        self.log_test("Create New User", success)
        return success

    def test_get_user_courses(self):
        """Test get user courses"""
        if not self.test_user_id:
            self.log_test("Get User Courses", False, "No test user ID available")
            return False
            
        success, response = self.make_request('GET', f'/api/user/{self.test_user_id}/courses')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                courses = data['data']
                print(f"   📚 User has {len(courses)} courses")
        self.log_test("Get User Courses", success)
        return success

    def test_get_user_models(self):
        """Test get user models"""
        if not self.test_user_id:
            self.log_test("Get User Models", False, "No test user ID available")
            return False
            
        success, response = self.make_request('GET', f'/api/user/{self.test_user_id}/models')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                models_data = data['data']
                trained = models_data.get('trained_models', [])
                untrained = models_data.get('untrained_models', [])
                print(f"   🤖 Trained: {len(trained)}, Untrained: {len(untrained)}")
        self.log_test("Get User Models", success)
        return success

    def test_add_user_courses(self):
        """Test adding courses to existing user"""
        if not self.test_user_id:
            self.log_test("Add User Courses", False, "No test user ID available")
            return False
            
        additional_courses = [6, 7, 8]
        success, response = self.make_request('POST', f'/api/user/{self.test_user_id}/courses/add',
                                            {'course_ids': additional_courses})
        if success:
            data = response.json()
            success = data.get('status') == 'success'
        self.log_test("Add User Courses", success)
        return success

    def test_train_model(self):
        """Test model training"""
        if not self.test_user_id:
            self.log_test("Train Model", False, "No test user ID available")
            return False
            
        model_name = "Course Similarity"
        success, response = self.make_request('POST', f'/api/user/{self.test_user_id}/train',
                                            {'user_id': self.test_user_id, 'model_name': model_name})
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'task_id' in data
            if success:
                self.task_id = data['task_id']
                print(f"   🔄 Training started with task ID: {self.task_id}")
        self.log_test("Train Model", success)
        return success

    def test_training_status(self):
        """Test training status check"""
        if not self.task_id:
            self.log_test("Training Status", False, "No task ID available")
            return False
            
        # Wait a bit for training to start
        time.sleep(2)
        
        success, response = self.make_request('GET', f'/api/training/{self.task_id}/status')
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                status_data = data['data']
                print(f"   📊 Training status: {status_data.get('status', 'unknown')}")
        self.log_test("Training Status", success)
        return success

    def test_predict_courses(self):
        """Test course prediction"""
        if not self.test_user_id:
            self.log_test("Predict Courses", False, "No test user ID available")
            return False
            
        # Wait for training to potentially complete
        time.sleep(5)
        
        model_name = "Course Similarity"
        success, response = self.make_request('POST', f'/api/user/{self.test_user_id}/predict',
                                            {'user_id': self.test_user_id, 'model_name': model_name, 'n_recommendations': 5})
        if success:
            data = response.json()
            success = data.get('status') == 'success' and 'data' in data
            if success:
                recommendations = data['data']
                print(f"   🎯 Generated {len(recommendations)} recommendations")
        self.log_test("Predict Courses", success)
        return success

    def test_invalid_endpoints(self):
        """Test error handling for invalid endpoints"""
        # Test invalid user ID
        success, response = self.make_request('GET', '/api/user/99999/courses', expected_status=500)
        success = not success  # We expect this to fail
        self.log_test("Invalid User ID Handling", success)
        
        # Test invalid model name
        if self.test_user_id:
            success, response = self.make_request('POST', f'/api/user/{self.test_user_id}/predict',
                                                {'user_id': self.test_user_id, 'model_name': 'Invalid Model'}, expected_status=400)
            success = not success  # We expect this to fail
            self.log_test("Invalid Model Name Handling", success)

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Course Recommendation API Tests")
        print("=" * 50)
        
        # Basic connectivity tests
        if not self.test_health_check():
            print("❌ API is not healthy, stopping tests")
            return False
            
        self.test_root_endpoint()
        
        # Data retrieval tests
        self.test_get_courses()
        self.test_get_ratings()
        self.test_get_users()
        
        # User management tests
        self.test_create_new_user()
        self.test_get_user_courses()
        self.test_get_user_models()
        self.test_add_user_courses()
        
        # ML functionality tests
        self.test_train_model()
        self.test_training_status()
        self.test_predict_courses()
        
        # Error handling tests
        self.test_invalid_endpoints()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed")
            return False

def main():
    """Main test execution"""
    print(f"🕐 Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = CourseRecommendationAPITester()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())