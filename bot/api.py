import os
import uuid
import asyncio
import shutil
from datetime import datetime
from typing import List, Optional
import tempfile
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from psycopg.errors import Error as PsycopgError
from pathlib import Path

from .database import (
    get_db, init_db, 
    # User operations
    create_user, get_user_by_username, get_user_by_email,
    # Document operations
    create_document, get_document, get_documents_by_user, update_document_processed, 
    update_document_error, delete_document,
    # Generation operations
    create_generation, get_generation, get_generations_by_user, update_generation_status,
    update_generation_counts, delete_generation,
    # Client type operations
    create_client_type, get_client_types_by_generation,
    # Analysis operations
    update_generation_analysis, complete_generation_analysis, update_generation_analysis_error
)
from .schemas import (
    UserCreate, UserResponse, DocumentResponse, GenerationCreate, 
    GenerationResponse, GenerationDetailResponse, AnalysisCreate, 
    AnalysisResponse, Token
)
from .auth import (
    get_current_active_user, get_password_hash, generate_api_key,
    authenticate_user, create_access_token, verify_api_key
)
from .utils import setup_logger
from .parser import parse_document
from .creator import create_prompts
from .analyzer import analyze_prompts

# Setup logger
logger = setup_logger("api")

# Create FastAPI app
app = FastAPI(
    title="BFSI Sales Bot Generator API",
    description="API for generating BFSI sales bot prompts based on knowledge base and agent persona",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key header for alternative authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize database
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")
    
    # Create data directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/prompts", exist_ok=True)
    os.makedirs("data/analysis", exist_ok=True)
    logger.info("Data directories created")


# Authentication routes
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    conn = Depends(get_db)
):
    user = authenticate_user(conn, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user['username']})
    logger.info(f"User {user['username']} logged in successfully")
    return {"access_token": access_token, "token_type": "bearer"}


# Helper function to get user from API key or JWT token
async def get_user_from_auth(
    api_key: str = Depends(API_KEY_HEADER),
    user = Depends(get_current_active_user),
    conn = Depends(get_db)
):
    # If JWT token is valid, return the user
    if user:
        return user
    
    # If API key is provided, verify it
    if api_key:
        api_user = verify_api_key(api_key)
        if api_user:
            return api_user
    
    # If neither authentication method works, raise 401
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# User routes
@app.post("/users", response_model=UserResponse)
async def create_user_route(user: UserCreate, conn = Depends(get_db)):
    db_user = get_user_by_username(user.username)
    if db_user:
        logger.warning(f"Attempted to create duplicate username: {user.username}")
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_email = get_user_by_email(user.email)
    if db_email:
        logger.warning(f"Attempted to create user with existing email: {user.email}")
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    password_hash = get_password_hash(user.password)
    
    try:
        new_user = create_user(user.username, user.email, password_hash)
        logger.info(f"New user created: {user.username}")
        return new_user
    except PsycopgError as e:
        logger.error(f"Database error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating user")


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user = Depends(get_user_from_auth)):
    return current_user


# Document routes
@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    document_type: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    # Validate document type
    if document_type not in ["knowledge_base", "agent_persona"]:
        raise HTTPException(status_code=400, detail="Invalid document type")
    
    # Create a unique filename to prevent overwrites
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("data/uploads", unique_filename)
    
    # Save the uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error saving file")
    
    # Create document record
    try:
        document = create_document(
            current_user['id'], 
            file.filename, 
            file_path, 
            document_type
        )
        
        logger.info(f"Document uploaded: {document['id']} - {document['filename']}")
        
        # Process document in background
        if background_tasks:
            background_tasks.add_task(
                process_document_task, document['id']
            )
        
        return document
    except PsycopgError as e:
        logger.error(f"Database error creating document: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating document record")


async def process_document_task(document_id: int):
    """Process document in background."""
    db_document = get_document(document_id)
    if not db_document:
        logger.error(f"Document not found: {document_id}")
        return
    
    try:
        # Construct a proper output path for the parsed document
        # e.g., data/parsed/user_<user_id>_doc_<document_id>_<filename>.md
        original_filename = os.path.basename(db_document['file_path'])
        parsed_filename = f"user_{db_document['user_id']}_doc_{document_id}_{original_filename}.md"
        parsed_output_dir = os.path.join("data", "parsed")
        os.makedirs(parsed_output_dir, exist_ok=True) # Ensure directory exists
        output_path = os.path.join(parsed_output_dir, parsed_filename)

        # Process document using parser with the correct output_path
        parse_document(db_document['file_path'], output_path)
        
        # Get content preview (strictly < 512 chars)
        content_preview = None
        preview_max_len = 500 # Max length for the content part
        try:
            with open(output_path, "r", encoding="utf-8") as f: # Added encoding
                content = f.read(preview_max_len)
            if len(content) >= preview_max_len:
                content_preview = content[:preview_max_len - 3] + "..." # Ensure total length is <= preview_max_len
            else:
                content_preview = content
        except Exception as e:
            logger.error(f"Error reading content preview from {output_path}: {str(e)}") # Log output_path
        
        # Update document record
        updated_doc = update_document_processed(document_id, output_path, content_preview)
        logger.info(f"Document processed successfully: {document_id}")
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        update_document_error(document_id, str(e))


@app.get("/documents", response_model=List[DocumentResponse])
async def get_documents(
    document_type: Optional[str] = None,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    if document_type and document_type not in ["knowledge_base", "agent_persona"]:
        raise HTTPException(status_code=400, detail="Invalid document type")
    
    documents = get_documents_by_user(current_user['id'], document_type)
    return documents


@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document_route(
    document_id: int,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    document = get_document(document_id, current_user['id'])
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@app.delete("/documents/{document_id}")
async def delete_document_route(
    document_id: int,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    # First, get the document to check if it exists and get file paths
    document = get_document(document_id, current_user['id'])
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove the file
        if os.path.exists(document['file_path']):
            os.remove(document['file_path'])
        
        # Remove the processed file if it exists
        if document.get('output_path') and os.path.exists(document['output_path']):
            os.remove(document['output_path'])
        
        # Delete document from database
        result = delete_document(document_id, current_user['id'])
        logger.info(f"Document deleted: {document_id}")
        
        return {"detail": "Document deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting document")


# Generation routes
@app.post("/generations", response_model=GenerationResponse)
async def create_generation_route(
    generation: GenerationCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    # Check if knowledge base and agent persona documents exist and belong to the user
    kb_doc = get_document(generation.knowledge_base_id, current_user['id'])
    persona_doc = get_document(generation.agent_persona_id, current_user['id'])
    
    if not kb_doc or kb_doc['document_type'] != "knowledge_base" or not kb_doc['processed']:
        raise HTTPException(
            status_code=404, 
            detail="Knowledge base document not found or not processed"
        )
    
    if not persona_doc or persona_doc['document_type'] != "agent_persona" or not persona_doc['processed']:
        raise HTTPException(
            status_code=404, 
            detail="Agent persona document not found or not processed"
        )
    
    # Create output directory
    output_dir = os.path.join("data/prompts", f"{current_user['username']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create generation record
        gen_record = create_generation(
            current_user['id'],
            generation.knowledge_base_id,
            generation.agent_persona_id,
            generation.questions_per_client,
            output_dir
        )
        
        logger.info(f"Generation task created: {gen_record['id']}")
        
        # Start generation in background
        background_tasks.add_task(
            process_generation_task,
            gen_record['id'], 
            kb_doc['output_path'], 
            persona_doc['output_path'],
            output_dir,
            generation.questions_per_client
        )
        
        return gen_record
    except PsycopgError as e:
        logger.error(f"Database error creating generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating generation task")


async def process_generation_task(
    generation_id: int,
    kb_path: str,
    persona_path: str,
    output_dir: str,
    questions_per_client: int
):
    """Process generation in background."""
    db_generation = get_generation(generation_id)
    if not db_generation:
        logger.error(f"Generation not found: {generation_id}")
        return
    
    # Fetch the username associated with this generation for logging purposes
    log_username = db_generation.get('username', f"user_{db_generation['user_id']}")

    try:
        # Update status to processing
        update_generation_status(generation_id, "processing")
        
        # Ensure output_dir (which is specific to this generation) is a Path object for creator
        generation_specific_output_dir = Path(output_dir)

        # Run creator script with the new signature
        client_type_objects = create_prompts(
            knowledge_base_path=kb_path,
            agent_persona_path=persona_path,
            questions_per_client=questions_per_client,
            output_dir=generation_specific_output_dir,
            username_for_logging=log_username
        )
        
        if not client_type_objects:
            logger.error(f"create_prompts returned no client types for generation_id: {generation_id}")
            update_generation_status(generation_id, "failed", "Failed to generate client types in creator module")
            return

        # Create client type records and count questions
        client_type_count = 0
        question_count = 0
        
        for client_obj in client_type_objects:
            client_output_filename = generation_specific_output_dir / f"{client_obj.client_type}_prompt.txt"

            create_client_type(
                generation_id,
                client_obj.client_type,
                client_obj.description,
                len(getattr(client_obj, 'questions', [])),
                str(client_output_filename)
            )
            client_type_count += 1
        
        # Update generation status to completed
        update_generation_status(generation_id, "completed")
        logger.info(f"Generation task marked completed for: {generation_id}. Detailed client type/question counts need to be confirmed.")

    except Exception as e:
        # Handle errors
        logger.error(f"Error in generation task {generation_id}: {str(e)}", exc_info=True)
        update_generation_status(generation_id, "failed", str(e))


@app.get("/generations", response_model=List[GenerationResponse])
async def get_generations_route(
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    generations = get_generations_by_user(current_user['id'])
    return generations


@app.get("/generations/{generation_id}", response_model=GenerationDetailResponse)
async def get_generation_route(
    generation_id: int,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    generation = get_generation(generation_id, current_user['id'])
    
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    # Get client types
    client_types = get_client_types_by_generation(generation_id)
    
    # Create a GenerationDetailResponse
    response = {**generation, "client_types": client_types}
    
    return response


@app.delete("/generations/{generation_id}")
async def delete_generation_route(
    generation_id: int,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    generation = get_generation(generation_id, current_user['id'])
    
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    try:
        # Remove output directory and files
        if generation['output_directory'] and os.path.exists(generation['output_directory']):
            shutil.rmtree(generation['output_directory'])
        
        # Remove analysis file if it exists
        if generation.get('analysis_path') and os.path.exists(generation['analysis_path']):
            os.remove(generation['analysis_path'])
        
        # Delete generation from database (this also deletes client types due to cascade)
        result = delete_generation(generation_id, current_user['id'])
        
        logger.info(f"Generation deleted: {generation_id}")
        
        return {"detail": "Generation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting generation {generation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error deleting generation")


# Analysis routes
@app.post("/analysis", response_model=AnalysisResponse)
async def create_analysis(
    analysis: AnalysisCreate,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    # Check if generation exists and belongs to the user
    generation = get_generation(analysis.generation_id, current_user['id'])
    
    if not generation or generation['status'] != "completed":
        raise HTTPException(
            status_code=404, 
            detail="Generation not found or not completed"
        )
    
    if generation['analysis_completed']:
        # Return existing analysis
        return {
            "generation_id": generation['id'],
            "analysis_path": generation['analysis_path'],
            "completed_at": generation['analysis_completed_at']
        }
    
    # Create analysis output path
    analysis_dir = os.path.join("data/analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    analysis_path = os.path.join(
        analysis_dir, 
        f"analysis_{current_user['username']}_{generation['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    try:
        # Update generation record
        updated = update_generation_analysis(generation['id'], analysis_path)
        
        # Start analysis in background
        background_tasks.add_task(
            process_analysis_task,
            generation['id'],
            generation['output_directory'],
            analysis_path
        )
        
        return {
            "generation_id": generation['id'],
            "analysis_path": analysis_path,
            "completed_at": None
        }
    except PsycopgError as e:
        logger.error(f"Database error creating analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Error creating analysis task")


async def process_analysis_task(
    generation_id: int,
    prompts_dir: str,
    analysis_path: str
):
    """Process analysis in background."""
    generation = get_generation(generation_id)
    if not generation:
        logger.error(f"Generation not found: {generation_id}")
        return
    
    try:
        # Run analyzer script
        analyze_prompts(prompts_dir, analysis_path)
        
        # Update generation record
        complete_generation_analysis(generation_id)
        
        logger.info(f"Analysis task completed for generation: {generation_id}")
    except Exception as e:
        # Handle errors
        logger.error(f"Error in analysis task for generation {generation_id}: {str(e)}")
        update_generation_analysis_error(generation_id, str(e))


@app.get("/analysis/{generation_id}")
async def get_analysis(
    generation_id: int,
    current_user = Depends(get_user_from_auth),
    conn = Depends(get_db)
):
    # Check if generation exists and belongs to the user
    generation = get_generation(generation_id, current_user['id'])
    
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    
    if not generation['analysis_completed'] or not generation.get('analysis_path'):
        raise HTTPException(status_code=404, detail="Analysis not found or not completed")
    
    if not os.path.exists(generation['analysis_path']):
        raise HTTPException(status_code=404, detail="Analysis file not found")
    
    # Return the analysis file as a response
    return FileResponse(
        generation['analysis_path'],
        media_type="text/markdown",
        filename=os.path.basename(generation['analysis_path'])
    ) 