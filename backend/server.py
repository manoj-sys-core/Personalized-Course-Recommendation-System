from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import sys
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from dotenv import load_dotenv

# Import the existing ML backend functions
import ml_backend

# Load environment variables
load_dotenv()

# Global variable to store training status
training_status = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Course Recommendation API starting up...")
    yield
    # Shutdown
    print("📴 Course Recommendation API shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Course Recommendation API",
    description="API for Personalized Course Recommendation System with ML Models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CourseSelection(BaseModel):
    course_ids: List[int]

class NewUserRequest(BaseModel):
    course_ids: List[int]

class ExistingUserRequest(BaseModel):
    user_id: int
    additional_courses: Optional[List[int]] = []

class TrainModelRequest(BaseModel):
    user_id: int
    model_name: str

class PredictRequest(BaseModel):
    user_id: int
    model_name: str
    n_recommendations: Optional[int] = 10

# API Routes

@app.get("/")
async def root():
    return {"message": "Course Recommendation API is running!", "status": "healthy"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/api/courses")
async def get_courses():
    """Get all available courses"""
    try:
        courses_df = ml_backend.load_course()
        courses = courses_df.to_dict('records')
        return {"status": "success", "data": courses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading courses: {str(e)}")

@app.get("/api/ratings")
async def get_ratings():
    """Get all ratings data"""
    try:
        ratings_df = ml_backend.load_rating()
        ratings = ratings_df.to_dict('records')
        return {"status": "success", "data": ratings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading ratings: {str(e)}")

@app.get("/api/user/{user_id}/courses")
async def get_user_courses(user_id: int):
    """Get courses completed by a specific user"""
    try:
        ratings_df = ml_backend.load_rating()
        courses_df = ml_backend.load_course()
        
        user_courses = ratings_df[ratings_df['user'] == user_id]
        enrolled_ids = user_courses['item'].unique().tolist()
        enrolled_courses = courses_df[courses_df['COURSE_ID'].isin(enrolled_ids)]
        
        courses = enrolled_courses.to_dict('records')
        return {"status": "success", "data": courses}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading user courses: {str(e)}")

@app.get("/api/user/{user_id}/models")
async def get_user_models(user_id: int):
    """Get trained models for a specific user"""
    try:
        model_map_df = ml_backend.load_user_model_map_by_userid(user_id)
        trained_models = model_map_df['model'].tolist() if 'model' in model_map_df.columns else []
        
        all_models = [
            "Course Similarity",
            "User Profile", 
            "Clustering",
            "Clustering with PCA",
            "Neural Network",
            "Regression with Embedding Features",
            "Classification with Embedding Features"
        ]
        
        untrained_models = [m for m in all_models if m not in trained_models]
        
        return {
            "status": "success", 
            "data": {
                "trained_models": trained_models,
                "untrained_models": untrained_models
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading user models: {str(e)}")

@app.post("/api/user/new")
async def create_new_user(request: NewUserRequest):
    """Create a new user with selected courses"""
    try:
        # Load existing ratings to get the next user ID
        ratings_df = ml_backend.load_rating()
        new_user_id = int(ratings_df['user'].max()) + 1
        
        # Create new ratings for the user
        from supabase import create_client
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        supabase = create_client(url, key)
        
        new_rows = [
            {"user": new_user_id, "item": course_id, "rating": 3}
            for course_id in request.course_ids
        ]
        
        insert_response = supabase.table("Ratings").insert(new_rows).execute()
        
        if insert_response.data:
            return {
                "status": "success", 
                "message": f"New user created successfully",
                "data": {"user_id": new_user_id}
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to insert ratings")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating new user: {str(e)}")

@app.post("/api/user/{user_id}/courses/add")
async def add_user_courses(user_id: int, request: CourseSelection):
    """Add additional courses for an existing user"""
    try:
        from supabase import create_client
        url = os.getenv('SUPABASE_URL')
        key = os.getenv('SUPABASE_KEY')
        supabase = create_client(url, key)
        
        # Add new courses
        new_rows = [
            {"user": user_id, "item": course_id, "rating": 3}
            for course_id in request.course_ids
        ]
        
        insert_response = supabase.table("Ratings").insert(new_rows).execute()
        
        if insert_response.data:
            # Clear existing trained models for this user
            supabase.table("User_Model_Map").delete().eq("userid", user_id).execute()
            
            return {
                "status": "success",
                "message": "Courses added successfully. Previous models cleared for retraining."
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add courses")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding courses: {str(e)}")

def train_model_background(user_id: int, model_name: str, task_id: str):
    """Background task for model training"""
    try:
        training_status[task_id] = {"status": "training", "progress": "Starting training..."}
        
        if model_name == "Course Similarity":
            result = ml_backend.course_similarity_train()
        elif model_name == "User Profile":
            result = ml_backend.user_profile_train()
        elif model_name in ["Clustering", "Clustering with PCA"]:
            result = ml_backend.kMeans_train(model_name)
        elif model_name == "Neural Network":
            result = ml_backend.NCF_train()
        elif model_name in ["Regression with Embedding Features", "Classification with Embedding Features"]:
            result = ml_backend.Embedding_train(model_name)
        else:
            result = f"❌ Training logic not implemented for {model_name}"
        
        if result.startswith("✅"):
            # Add entry to User_Model_Map
            from supabase import create_client
            url = os.getenv('SUPABASE_URL')
            key = os.getenv('SUPABASE_KEY')
            supabase = create_client(url, key)
            
            supabase.table("User_Model_Map").insert({
                "userid": user_id, 
                "model": model_name
            }).execute()
            
            training_status[task_id] = {"status": "completed", "result": result}
        else:
            training_status[task_id] = {"status": "failed", "result": result}
            
    except Exception as e:
        training_status[task_id] = {"status": "failed", "result": f"❌ Training failed: {str(e)}"}

@app.post("/api/user/{user_id}/train")
async def train_model(user_id: int, request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Train a model for a specific user"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Start background training task
    background_tasks.add_task(train_model_background, user_id, request.model_name, task_id)
    
    return {
        "status": "success",
        "message": f"Training started for {request.model_name}",
        "task_id": task_id
    }

@app.get("/api/training/{task_id}/status")
async def get_training_status(task_id: str):
    """Get training status for a task"""
    if task_id in training_status:
        return {"status": "success", "data": training_status[task_id]}
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.post("/api/user/{user_id}/predict")
async def predict_courses(user_id: int, request: PredictRequest):
    """Get course recommendations for a user"""
    try:
        model_name = request.model_name
        n_rec = request.n_recommendations
        
        if model_name == "Course Similarity":
            prediction_df = ml_backend.course_similarity_predict(user_id)
        elif model_name == "User Profile":
            prediction_df = ml_backend.user_profile_predict(user_id)
        elif model_name in ["Clustering", "Clustering with PCA"]:
            prediction_df = ml_backend.kMeans_pred(user_id, model_name)
        elif model_name == "Neural Network":
            prediction_df = ml_backend.NCF_predict(user_id)
        elif model_name in ["Regression with Embedding Features", "Classification with Embedding Features"]:
            prediction_df = ml_backend.Embedding_Predict(user_id, model_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")
        
        if not prediction_df.empty:
            recommendations = prediction_df.head(n_rec).to_dict('records')
            return {"status": "success", "data": recommendations}
        else:
            return {"status": "success", "data": [], "message": "No recommendations available"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@app.get("/api/users")
async def get_all_users():
    """Get list of all user IDs"""
    try:
        ratings_df = ml_backend.load_rating()
        user_ids = sorted(ratings_df['user'].unique().tolist())
        return {"status": "success", "data": user_ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading users: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )