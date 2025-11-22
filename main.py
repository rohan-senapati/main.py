import json
import os
import http.client
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from dotenv import load_dotenv
import uuid
import jwt
from fastapi.security.http import HTTPAuthorizationCredentials
from database import get_db, SECRET_KEY, ALGORITHM, REFRESH_TOKEN_EXPIRE_DAYS, ACCESS_TOKEN_EXPIRE_MINUTES
from models import (UserDB, SignUpRequest, TokenResponse, UserResponse, LoginRequest, RefreshTokenRequest,
                    UpdateProfileRequest, ChangePasswordRequest, CourseRecommendationResponse, JobSearchRequest,
                    CourseNameRequest, AvailableCoursesResponse, CourseDetailsResponse)
from typing import List, Optional

load_dotenv()

# FastAPI app
app = FastAPI(title="Career Navigator API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
security = HTTPBearer()

# ==================== GLOBAL VARIABLES ====================
course_data: Optional[pd.DataFrame] = None
course_vectorizer = None
course_similarity_matrix = None


# ==================== AUTH FUNCTIONS ====================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: str, expires_delta: timedelta | None = None):
    to_encode = {"user_id": user_id, "type": "access"}
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: str):
    to_encode = {"user_id": user_id, "type": "refresh"}
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    user_id = verify_token(token)
    user = db.query(UserDB).filter(UserDB.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


# ==================== COURSE RECOMMENDATION FUNCTIONS ====================
def load_course_data():
    """Load and prepare course data from CSV"""
    global course_data, course_vectorizer, course_similarity_matrix

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'Coursera.csv')
        model_dir = os.path.join(base_dir, 'model')
        vectorizer_path = os.path.join(model_dir, 'count_vectorizer.pkl')
        similarity_path = os.path.join(model_dir, 'cosine_similarity.npy')

        # Load CSV
        course_data = pd.read_csv(data_path)

        # Clean the data
        course_data['Course Name'] = course_data['Course Name'].fillna('')
        course_data['Course URL'] = course_data['Course URL'].fillna('')
        course_data['Course Rating'] = pd.to_numeric(course_data['Course Rating'], errors='coerce').fillna(0.0)
        course_data['Difficulty Level'] = course_data['Difficulty Level'].fillna('Not Specified')
        course_data['University'] = course_data['University'].fillna('Unknown')

        # Remove duplicates
        course_data = course_data.drop_duplicates(subset=['Course Name'], keep='first')

        # Try to load pre-computed models
        if os.path.exists(vectorizer_path) and os.path.exists(similarity_path):
            course_vectorizer = joblib.load(vectorizer_path)
            course_similarity_matrix = np.load(similarity_path)
            print("✅ Loaded pre-computed models")
        else:
            # Compute new models
            print("⚙️ Computing similarity matrix...")
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            course_data['combined_features'] = (
                    course_data['Course Name'].fillna('') + ' ' +
                    course_data['Difficulty Level'].fillna('') + ' ' +
                    course_data['University'].fillna('')
            )

            course_vectorizer = CountVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            count_matrix = course_vectorizer.fit_transform(course_data['combined_features'])
            course_similarity_matrix = cosine_similarity(count_matrix)

            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(course_vectorizer, vectorizer_path)
            np.save(similarity_path, course_similarity_matrix)
            print("✅ Computed and saved models")

        print(f"✅ Loaded {len(course_data)} courses")

    except Exception as e:
        print(f"❌ Error loading course data: {str(e)}")
        raise


def verify_csv_columns():
    """Verify CSV has the correct columns"""
    if course_data is None:
        return False

    required_columns = ['Course Name', 'Course URL', 'Course Rating',
                        'Difficulty Level', 'University']

    missing_columns = [col for col in required_columns if col not in course_data.columns]

    if missing_columns:
        print(f"❌ Missing columns in CSV: {missing_columns}")
        print(f"📋 Available columns: {course_data.columns.tolist()}")
        return False

    print(f"✅ All required columns present")
    return True


def find_course_index(course_name: str) -> Optional[int]:
    """Find the index of a course by name with improved matching"""
    if course_data is None:
        return None

    course_name_clean = course_name.strip()

    # Exact match (case-insensitive)
    exact_match = course_data[course_data['Course Name'].str.lower() == course_name_clean.lower()]
    if not exact_match.empty:
        print(f"✅ Found exact match for: {course_name_clean}")
        return exact_match.index[0]

    # Partial match (case-insensitive)
    partial_match = course_data[
        course_data['Course Name'].str.contains(
            course_name_clean,
            case=False,
            na=False,
            regex=False
        )
    ]
    if not partial_match.empty:
        print(f"✅ Found partial match for: {course_name_clean}")
        return partial_match.index[0]

    print(f"❌ No match found for: {course_name_clean}")
    return None


def get_course_recommendations(course_name: str, num_recommendations: int = 5) -> List[CourseRecommendationResponse]:
    """Get course recommendations"""
    if course_data is None or course_similarity_matrix is None:
        raise HTTPException(status_code=500, detail="Course data not loaded")

    course_idx = find_course_index(course_name)

    if course_idx is None:
        # Get better suggestions
        suggestions = []
        words = course_name.lower().split()
        if words:
            first_word = words[0]
            suggestions = course_data[
                course_data['Course Name'].str.lower().str.contains(
                    first_word,
                    case=False,
                    na=False,
                    regex=False
                )
            ]['Course Name'].head(5).tolist()

        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Course '{course_name}' not found",
                "suggestions": suggestions if suggestions else [
                    "Python Programming Essentials",
                    "Machine Learning",
                    "Business Strategy"
                ],
                "hint": "Try selecting from the dropdown list"
            }
        )

    similarity_scores = list(enumerate(course_similarity_matrix[course_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    recommendations = []
    for idx, score in similarity_scores:
        course = course_data.iloc[idx]
        rating = float(course['Course Rating']) if pd.notna(course['Course Rating']) else 0.0

        recommendations.append(
            CourseRecommendationResponse(
                name=str(course['Course Name']),
                url=str(course['Course URL']),
                rating=rating,
                difficulty=str(course['Difficulty Level'])
            )
        )

    return recommendations


# ==================== JOB SEARCH FUNCTIONS ====================
def search_jobs_api(term: str, location: str):
    """Search for jobs using the Jobs API"""
    try:
        conn = http.client.HTTPSConnection("jobs-search-api.p.rapidapi.com")
        payload = json.dumps({
            "search_term": term,
            "location": location,
            "results_wanted": 5,
            "site_name": ["indeed", "linkedin", "zip_recruiter", "glassdoor"],
            "distance": 50,
            "job_type": "fulltime",
            "is_remote": False,
            "linkedin_fetch_description": False,
            "hours_old": 72
        })
        headers = {
            'x-rapidapi-key': os.getenv("RAPIDAPI_KEY", "b4e2724337mshafbcc730f5432cep11eec7jsncf31014cd620"),
            'x-rapidapi-host': "jobs-search-api.p.rapidapi.com",
            'Content-Type': "application/json"
        }
        conn.request("POST", "/getjobs", payload, headers)
        res = conn.getresponse()
        data = res.read()
        decoded_data = data.decode("utf-8")
        job_data = json.loads(decoded_data)

        # Get jobs from response
        jobs = job_data.get("jobs", [])

        # Process jobs to ensure they have the required fields
        processed_jobs = []
        for job in jobs:
            # Ensure job_url is present and valid
            job_url = job.get('job_url') or job.get('url') or job.get('link')

            # Only include jobs with valid URLs
            if job_url and job_url.startswith('http'):
                processed_job = {
                    'title': job.get('title') or job.get('job_title') or 'No Title',
                    'company': job.get('company') or job.get('company_name') or 'No Company',
                    'location': job.get('location') or job.get('job_location') or location,
                    'description': job.get('description') or job.get('job_description') or 'No description available',
                    'job_url': job_url,
                    'employment_type': job.get('employment_type') or job.get('job_type') or 'Full-time',
                    'site': job.get('site') or 'Job Site',
                    'salary_range': job.get('salary_range') or job.get('salary'),
                    'posted_date': job.get('posted_date') or job.get('date_posted')
                }
                processed_jobs.append(processed_job)

        return processed_jobs

    except Exception as e:
        print(f"Error searching jobs: {str(e)}")
        return []


# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    """Load course data on startup"""
    load_course_data()
    verify_csv_columns()


# ==================== API ENDPOINTS ====================
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Career Navigator API",
        "status": "running",
        "version": "2.0.0",
        "total_courses": len(course_data) if course_data is not None else 0
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "course_data_loaded": course_data is not None,
        "total_courses": len(course_data) if course_data is not None else 0,
        "similarity_matrix_loaded": course_similarity_matrix is not None,
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== AUTH ENDPOINTS ====================
@app.post("/auth/signup", response_model=TokenResponse, tags=["Authentication"])
async def signup(request: SignUpRequest, db: Session = Depends(get_db)):
    existing_user = db.query(UserDB).filter(UserDB.email == request.email).first()
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    new_user = UserDB(
        id=str(uuid.uuid4()),
        email=request.email,
        first_name=request.first_name,
        last_name=request.last_name,
        password_hash=hash_password(request.password),
        phone_number=request.phone_number,
        date_of_birth=request.date_of_birth,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(new_user.id)
    refresh_token = create_refresh_token(new_user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.from_orm(new_user),
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == request.email).first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=UserResponse.from_orm(user),
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    user_id = verify_token(request.refresh_token)
    user = db.query(UserDB).filter(UserDB.id == user_id).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    access_token = create_access_token(user.id)
    new_refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=UserResponse.from_orm(user),
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.post("/auth/change-password", tags=["Authentication"])
async def change_password(
        request: ChangePasswordRequest,
        current_user: UserDB = Depends(get_current_user),
        db: Session = Depends(get_db),
):
    if not verify_password(request.current_password, current_user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

    current_user.password_hash = hash_password(request.new_password)
    db.commit()

    return {"message": "Password changed successfully"}


# ==================== USER ENDPOINTS ====================
@app.get("/users/me", response_model=UserResponse, tags=["Users"])
async def get_current_user_profile(current_user: UserDB = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)


@app.put("/users/me", response_model=UserResponse, tags=["Users"])
async def update_profile(
        request: UpdateProfileRequest,
        current_user: UserDB = Depends(get_current_user),
        db: Session = Depends(get_db),
):
    update_data = request.dict(exclude_unset=True)

    for field, value in update_data.items():
        setattr(current_user, field, value)

    current_user.updated_at = datetime.utcnow()

    if (current_user.first_name and current_user.last_name and
            current_user.current_role and current_user.education_level):
        current_user.is_profile_complete = True

    db.commit()
    db.refresh(current_user)

    return UserResponse.from_orm(current_user)


@app.delete("/users/me", tags=["Users"])
async def delete_account(
        current_user: UserDB = Depends(get_current_user),
        db: Session = Depends(get_db),
):
    db.delete(current_user)
    db.commit()

    return {"message": "Account deleted successfully"}


# ==================== JOB SEARCH ENDPOINTS ====================
@app.post("/search/jobs", tags=["Job Search"])
async def search_jobs(request: JobSearchRequest):
    """Search for jobs based on term and location"""
    jobs = search_jobs_api(request.term, request.location)
    return {"jobs": jobs}


# ==================== COURSE RECOMMENDATION ENDPOINTS ====================
@app.post("/recommend/courses", response_model=List[CourseRecommendationResponse], tags=["Course Recommendations"])
async def recommend_courses(request: CourseNameRequest):
    """Get course recommendations based on a course name"""
    try:
        print(f"🔍 Searching for course: '{request.course_name}'")

        if course_data is None or course_similarity_matrix is None:
            raise HTTPException(status_code=500, detail="Course data not loaded")

        recommendations = get_course_recommendations(request.course_name, num_recommendations=5)

        print(f"✅ Found {len(recommendations)} recommendations")
        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in recommend_courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/courses/available", response_model=AvailableCoursesResponse, tags=["Course Recommendations"])
async def get_available_courses(
        limit: Optional[int] = 1000,
        difficulty: Optional[str] = None,
        university: Optional[str] = None,
        min_rating: Optional[float] = None
):
    """Get list of available courses with optional filtering"""
    try:
        if course_data is None:
            raise HTTPException(status_code=500, detail="Course data not loaded")

        filtered_data = course_data.copy()

        if difficulty:
            filtered_data = filtered_data[
                filtered_data['Difficulty Level'].str.lower() == difficulty.lower()
                ]

        if university:
            filtered_data = filtered_data[
                filtered_data['University'].str.contains(university, case=False, na=False, regex=False)
            ]

        if min_rating is not None:
            filtered_data = filtered_data[
                pd.to_numeric(filtered_data['Course Rating'], errors='coerce') >= min_rating
                ]

        courses = sorted(filtered_data['Course Name'].unique().tolist())[:limit]

        print(f"✅ Returning {len(courses)} courses out of {len(filtered_data)} total")

        return AvailableCoursesResponse(
            courses=courses,
            total_count=len(filtered_data)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in get_available_courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching courses: {str(e)}")


@app.get("/courses/search", response_model=List[str], tags=["Course Recommendations"])
async def search_courses(query: str, limit: int = 20):
    """Search for courses by name"""
    try:
        if course_data is None:
            raise HTTPException(status_code=500, detail="Course data not loaded")

        if not query or len(query) < 2:
            return []

        results = course_data[
            course_data['Course Name'].str.contains(query, case=False, na=False, regex=False)
        ]['Course Name'].unique().tolist()[:limit]

        print(f"✅ Found {len(results)} courses matching '{query}'")
        return results

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in search_courses: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching courses: {str(e)}")


@app.get("/courses/details/{course_name}", response_model=CourseDetailsResponse, tags=["Course Recommendations"])
async def get_course_details(course_name: str):
    """Get detailed information about a specific course"""
    try:
        if course_data is None:
            raise HTTPException(status_code=500, detail="Course data not loaded")

        course_idx = find_course_index(course_name)

        if course_idx is None:
            raise HTTPException(
                status_code=404,
                detail=f"Course '{course_name}' not found"
            )

        course = course_data.iloc[course_idx]
        rating = float(course['Course Rating']) if pd.notna(course['Course Rating']) else 0.0

        return CourseDetailsResponse(
            name=str(course['Course Name']),
            url=str(course['Course URL']),
            rating=rating,
            difficulty=str(course['Difficulty Level']),
            university=str(course['University'])
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in get_course_details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching course details: {str(e)}")


@app.get("/courses/statistics", tags=["Course Recommendations"])
async def get_course_statistics():
    """Get statistics about the course catalog"""
    if course_data is None:
        raise HTTPException(status_code=500, detail="Course data not loaded")

    return {
        "total_courses": len(course_data),
        "universities": int(course_data['University'].nunique()),
        "difficulty_distribution": course_data['Difficulty Level'].value_counts().to_dict(),
        "average_rating": float(course_data['Course Rating'].mean()),
        "top_universities": course_data['University'].value_counts().head(10).to_dict()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)