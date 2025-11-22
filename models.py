import uuid
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator, field_validator
from sqlalchemy import Column, String, DateTime, Boolean
from database import Base, engine
from typing import List


#==================== Database Models ====================
class UserDB(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    phone_number = Column(String, nullable=True)
    date_of_birth = Column(DateTime, nullable=False)
    profile_picture = Column(String, nullable=True)
    bio = Column(String, nullable=True)
    location = Column(String, nullable=True)
    current_role = Column(String, nullable=True)
    company = Column(String, nullable=True)
    experience_years = Column(String, default="0")
    education_level = Column(String, nullable=True)
    is_email_verified = Column(Boolean, default=False)
    is_profile_complete = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ==================== Pydantic Models ====================
class SignUpRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    date_of_birth: datetime
    phone_number: str | None = None

    @field_validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if len(v) > 50:
            raise ValueError('Password must be at most 50 characters')
        return v

    @field_validator('first_name', 'last_name')
    def validate_names(cls, v):
        if not v or len(v) > 50:
            raise ValueError('Name must be between 1 and 50 characters')
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    phone_number: str | None
    date_of_birth: datetime
    profile_picture: str | None
    bio: str | None
    location: str | None
    current_role: str | None
    company: str | None
    experience_years: str
    education_level: str | None
    is_email_verified: bool
    is_profile_complete: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    bio: str | None = None
    location: str | None = None
    current_role: str | None = None
    company: str | None = None
    experience_years: str | None = None
    education_level: str | None = None
    phone_number: str | None = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

# Used in /search/jobs request
class JobSearchRequest(BaseModel):
    term: str
    location: str

# Used in /recommend/courses request
class CourseNameRequest(BaseModel):
    course_name: str

# Used in /recommend/courses response
class CourseRecommendationResponse(BaseModel):
    name: str
    url: str
    rating: float
    difficulty: str

# Used in /courses/available response
class AvailableCoursesResponse(BaseModel):
    courses: List[str]
    total_count: int

# Used in /courses/details/{course_name} response
class CourseDetailsResponse(BaseModel):
    name: str
    url: str
    rating: float
    difficulty: str
    university: str

