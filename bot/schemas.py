from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator


# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str

    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john.doe@example.com",
                "password": "securepassword123"
            }
        }


class UserResponse(UserBase):
    id: int
    created_at: datetime
    is_active: bool
    api_key: str

    class Config:
        orm_mode = True


# Document schemas
class DocumentBase(BaseModel):
    document_type: str

    @validator('document_type')
    def validate_document_type(cls, v):
        allowed_types = ["knowledge_base", "agent_persona"]
        if v not in allowed_types:
            raise ValueError(f"document_type must be one of {allowed_types}")
        return v


class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    filename: str
    uploaded_at: datetime
    processed: bool
    processed_at: Optional[datetime] = None
    content_preview: Optional[str] = None

    class Config:
        orm_mode = True


# Generation schemas
class GenerationCreate(BaseModel):
    knowledge_base_id: int
    agent_persona_id: int
    questions_per_client: Optional[int] = 50


class GenerationResponse(BaseModel):
    id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    client_types_count: int
    questions_count: int
    questions_per_client: int
    output_directory: Optional[str] = None
    error_message: Optional[str] = None
    analysis_path: Optional[str] = None
    analysis_completed: bool

    class Config:
        orm_mode = True


class ClientTypeResponse(BaseModel):
    id: int
    name: str
    description: str
    created_at: datetime
    question_count: int
    output_file: Optional[str] = None

    class Config:
        orm_mode = True


class GenerationDetailResponse(GenerationResponse):
    client_types: List[ClientTypeResponse] = []

    class Config:
        orm_mode = True


# Analysis schemas
class AnalysisCreate(BaseModel):
    generation_id: int


class AnalysisResponse(BaseModel):
    generation_id: int
    analysis_path: str
    completed_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Token schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None 